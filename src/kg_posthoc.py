"""Post-hoc knowledge-graph evaluation of simulated diagnostic dialogues (runs after kg_eval.py).

Steps
  check       LLM check of the true diagnosis' HPO phenotypes in each dialogue (quote-verified)
  rescore     recompute kg_eval metrics with step 0 + recovered phenotypes; add truth-directed metrics
  robustness  recompute the paper metrics under alternative likelihood parameters
  plot        per-step trajectories of the paper metrics for one case

Usage
  python kg_posthoc.py check      --model glm53
  python kg_posthoc.py rescore    --model glm53
  python kg_posthoc.py robustness --model glm53 [--no-eig] [--n-cases 100]
  python kg_posthoc.py plot       --model glm53 --case "cases_full_psychiatry/Case 22-2025"

Files (per model M)
  in:   kg_eval_M.json                                    (kg_eval.py)
        referrals.json   {case: {"presentation": referral line shown to the doctor}}
  out:  step0_cache.json                                  (shared across models)
        dx_check_M.json                                   (check cache)
        kg_eval_M_final.json/.csv, kg_eval_M_final_progress.jsonl
        robustness_M.csv, robustness_M_summary.csv, trajectory_<case>.png
"""

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import kg_eval as k

LLM = "genai:gemini-3.5-flash-lite"           # extraction, linking, check
CHECK_BATCH = 10                               # phenotypes per check call
VERIFY_BATCH = 20                              # (phenotype, quote) pairs per verification call
GENERIC = 0.05                                 # phenotype "generic" if annotated in >5% of diseases

REFERRALS = "referrals.json"
STEP0_CACHE = "step0_cache.json"

BASE_PARAMS = dict(P_HIT=0.6, P_BACKGROUND=0.1, P_EXCLUDED=0.02, P_PARTIAL=0.3, ABSENT_WEIGHT=0.3)
VARIANTS = {
    "baseline": {},
    "p_hit=0.4": {"P_HIT": 0.4},
    "p_hit=0.8": {"P_HIT": 0.8},
    "p_bg=0.05": {"P_BACKGROUND": 0.05},
    "p_bg=0.2": {"P_BACKGROUND": 0.2},
    "p_exc=0.005": {"P_EXCLUDED": 0.005},
    "no partial": {"P_PARTIAL": 0.0},
    "absent weight=1": {"ABSENT_WEIGHT": 1.0},
    "absent weight=0.1": {"ABSENT_WEIGHT": 0.1},
}
PAPER_METRICS = ["min_true_rank", "final_log_odds_true", "info_true_bits",
                 "mean_info_gain_bits", "mapped_fraction", "eig_true_ratio"]
ORIG_COLUMNS = ["final_true_rank", "min_true_rank", "final_log_odds_true", "error_type",
                "mapped_fraction", "mean_info_gain_bits", "mean_eig_ratio",
                "diagnostic_regret_bits", "total_info_bits"]

CHECK_PROMPT = """You read a doctor-patient encounter, split into numbered turns. Turn 0 is the referral
slip the doctor received before starting. For each phenotype in the list, decide whether the
encounter EXPLICITLY establishes that the patient has it. Be strict:
- Count it if the text states it, with the same meaning, a synonym, or lay wording that clearly
  means it ("tingling" = Paresthesia, "wiped out" = Fatigue).
- Do NOT infer. A lab value does not establish a named condition unless that condition is stated
  (hemoglobin 6.6 is "Anemia", not "Hemolytic anemia"; low phosphorus is not "rickets").
  A symptom does not establish its cause or its subtype ("arm jumping" is not "Focal-onset
  seizure" unless a seizure is stated). A suspected or "consistent with" finding does not
  establish a specific diagnosis.
- Do NOT count findings that are denied, absent, normal, past and resolved, or only mentioned in
  the doctor's question.
For each phenotype present give:
- "turn": the FIRST turn where it is established;
- "quote": a short VERBATIM quote (at most 20 words) from the patient's answer, the test result,
  or the referral slip of that turn, copied exactly.
When in doubt, leave it out.
Reply with JSON only: {"present": [{"id": "HP:...", "turn": 3, "quote": "..."}]}"""

VERIFY_PROMPT = """For each item you get a phenotype and a quote from a clinical encounter. Answer "yes"
only if the quote by itself explicitly establishes that the patient currently has that phenotype,
without inferring a cause, a subtype or a diagnosis that the quote does not state. Otherwise "no".
Reply with JSON only: {"answers": [{"item": 0, "ok": "yes"}]}"""


# --------------------------------------------------------------------------- io
def load_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def save_json(path, obj, indent=None):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def kg_cases(model):
    return load_json(f"kg_eval_{model}.json")["cases"]


def final_cases(model):
    rows = [x["row"] for x in read_jsonl(f"kg_eval_{model}_final_progress.jsonl")]
    rows = rows or load_json(f"kg_eval_{model}_final.json", {"cases": []})["cases"]
    return list({r["case_key"]: r for r in rows}.values())


def configure_llm(model=LLM):
    k.MODEL = model


def _words(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)      # [text](url) -> text
    text = re.sub(r"<[^>]+>", " ", text)
    return " " + " ".join(re.findall(r"[^\W_]+", text)) + " "


def is_verbatim(quote, text):
    q = _words(quote)
    return bool(q.strip()) and q in _words(text)


# ----------------------------------------------------------------- graph utils
def true_cols(kg, r):
    return [kg.col[x] for x in r["kg_diagnosis_ids"]
            if x in kg.col and kg.n_phenotypes(x) >= k.MIN_DX_PHENOTYPES]


def disease_phenotypes(kg, r):
    return sorted(set().union(*[set(kg.disease_terms(kg.col[x]))
                                for x in r["kg_diagnosis_ids"] if x in kg.col]))


def kl_bern(a, b):
    a, b = np.clip(a, 1e-9, 1 - 1e-9), np.clip(b, 1e-9, 1 - 1e-9)
    return a * np.log2(a / b) + (1 - a) * np.log2((1 - a) / (1 - b))


def new_findings(step, evidence):
    out = []
    for f in step["findings"]:
        if f.get("hpo"):
            g = frozenset(f["hpo"])
            if (g not in evidence or evidence[g]["status"] != f["status"]) and (g, f["status"]) not in out:
                out.append((g, f["status"]))
    return out


def truth_eig(kg, post, evidence, new, cols):
    """(EIG towards V of the best new finding, best EIG towards V available)."""
    v = cols[int(np.argmax(post[cols]))]
    eig = lambda g: float(kl_bern(kg.p_present(g)[v], kg.p_present(g) @ post))
    asked = max(eig(g) for g, _ in new)
    known = {t for g in evidence for t in g}
    top = np.argsort(-post, kind="stable")[:k.TOP_DISEASES_FOR_BEST]
    cand = {frozenset([t]) for c in list(top) + [v] for t in kg.disease_terms(int(c))
            if t not in known and kg.member(t).mean() <= k.MAX_TERM_PREVALENCE}
    return asked, max([asked] + [eig(g) for g in cand])


def trajectory(kg, r, with_eig=True):
    """Per-step metrics and their summary, from the HPO-linked findings saved in r['steps']."""
    cols = true_cols(kg, r)
    if not cols:
        return None
    evidence, logp, logp0, rows = {}, np.zeros(kg.n), None, []
    for s in r["steps"]:
        post = k._normalise(logp)
        h_before = k._entropy(post)
        new = new_findings(s, evidence)
        eig = rho = None
        if with_eig and new and s["kind"] != "given" and h_before >= k.SATURATED_BITS:
            eig, best = truth_eig(kg, post, evidence, new, cols)
            rho = eig / best if best > 1e-9 else None
        for g, st in new:
            evidence[g] = {"status": st, "step": s["step"], "weight": 1.0}
        if new:
            logp = k.log_likelihood(kg, evidence)
        post = k._normalise(logp)
        if s["kind"] == "given":
            logp0 = k.log2_mass(logp, cols)
            continue
        rows.append({"step": s["step"], "kind": s["kind"], "mapped": bool(new),
                     "ig": h_before - k._entropy(post) if new else None,
                     "rank": k.mid_rank(post, cols),
                     "log_odds": k.log_odds_vs_best_other(logp, cols),
                     "log_p_true": k.log2_mass(logp, cols), "eig_true": eig, "rho": rho})
    if logp0 is None:
        logp0 = float(np.log2(len(cols) / kg.n))
    for x in rows:
        x["info_true"] = x.pop("log_p_true") - logp0
    mean = lambda xs: round(float(np.mean(xs)), 4) if xs else None
    summary = {
        "min_true_rank": min((x["rank"] for x in rows), default=None),
        "final_log_odds_true": rows[-1]["log_odds"] if rows else None,
        "info_true_bits": round(rows[-1]["info_true"], 3) if rows else None,
        "mean_info_gain_bits": mean([x["ig"] for x in rows if x["ig"] is not None]),
        "mapped_fraction": round(sum(x["mapped"] for x in rows) / len(rows), 3) if rows else None,
        "eig_true_mean": mean([x["eig_true"] for x in rows if x["eig_true"] is not None]),
        "eig_true_ratio": mean([x["rho"] for x in rows if x["rho"] is not None]),
    }
    return {"steps": rows, "summary": summary}


# ------------------------------------------------------------------ 1. check
def referral_of(r, refs):
    given = [s["answer"] for s in r["steps"] if s["kind"] == "given" and s.get("answer")]
    return given[0] if given else (refs.get(r["case"]) or {}).get("presentation", "")


def encounter(r, refs):
    """{turn: quotable text}, full text for the LLM. Quotes only from answers/results/referral."""
    quotable, lines = {}, []
    ref = referral_of(r, refs)
    if ref:
        quotable[0] = ref
        lines.append(f"TURN 0 (referral slip): {ref}")
    for s in sorted(r["steps"], key=lambda s: s["step"]):
        if s["kind"] == "given":
            continue
        q, a = ("DOCTOR ASKS", "PATIENT") if s["kind"] == "ask" else ("DOCTOR ORDERS", "RESULT")
        quotable[s["step"]] = str(s["answer"])
        lines.append(f"TURN {s['step']}\n{q}: {s['action']}\n{a}: {s['answer']}")
    return quotable, "\n\n".join(lines)


def check_case(kg, r, refs):
    quotable, text = encounter(r, refs)
    phenotypes = disease_phenotypes(kg, r)
    found, bad_quote = {}, 0
    for _ in range(2):                                              # 2nd pass: only those not found
        left = sorted(set(phenotypes) - set(found))
        for i in range(0, len(left), CHECK_BATCH):
            batch = left[i:i + CHECK_BATCH]
            listing = "\n".join(f"{t} {kg.vocab.term_name(t)}" for t in batch)
            reply = k._llm(CHECK_PROMPT, f"PHENOTYPES:\n{listing}\n\nENCOUNTER:\n{text}")
            for p in reply.get("present") or []:
                if not isinstance(p, dict):
                    bad_quote += 1
                    continue
                pid, quote = p.get("id"), str(p.get("quote") or "")
                if pid not in batch or pid in found:
                    continue
                turns = [n for n, a in sorted(quotable.items()) if is_verbatim(quote, a)]
                if turns:
                    found[pid] = {"step": turns[0], "quote": quote}
                else:
                    bad_quote += 1
    ids, rejected = list(found), 0                                   # 2nd opinion on each quote
    for i in range(0, len(ids), VERIFY_BATCH):
        batch = ids[i:i + VERIFY_BATCH]
        listing = "\n".join(f'ITEM {j}: phenotype "{kg.vocab.term_name(p)}" | quote: "{found[p]["quote"]}"'
                            for j, p in enumerate(batch))
        reply = k._llm(VERIFY_PROMPT, listing)
        ok = {int(a["item"]): str(a.get("ok", "")).lower().startswith("y")
              for a in reply.get("answers") or []
              if isinstance(a, dict) and str(a.get("item", "")).lstrip("-").isdigit()}
        for j, p in enumerate(batch):
            if not ok.get(j, False):
                del found[p]
                rejected += 1
    return {"found": found, "rejected_quote": bad_quote, "rejected_verify": rejected,
            "has_step0": 0 in quotable}


def legacy_check(e):
    if "found" in e:
        return e
    return {"found": e.get("trovati", {}), "rejected_quote": e.get("scartati", 0),
            "rejected_verify": e.get("respinti_verifica", 0), "has_step0": e.get("step0", False)}


def check(model, cache_path=None, llm=LLM):
    configure_llm(llm)
    kg = k.load_graph(k.HPOVocab(), "hpo")
    refs = load_json(REFERRALS, {})
    cache_path = cache_path or f"dx_check_{model}.json"
    cache = load_json(cache_path, {})
    cases = kg_cases(model)
    for i, r in enumerate(cases, 1):
        if r["case_key"] in cache:
            continue
        try:
            cache[r["case_key"]] = check_case(kg, r, refs)
        except k.MappingFailed as e:
            print(f"{i}/{len(cases)} {r['case']}: LLM error ({e}), retried next run")
            continue
        save_json(cache_path, cache)
        e = cache[r["case_key"]]
        print(f"{i}/{len(cases)} {r['case']}: found {len(e['found'])} | "
              f"rejected quote {e['rejected_quote']} | rejected verify {e['rejected_verify']}")


def dx_columns(kg, r, entry):
    """dx_* columns and annotated found phenotypes for one case."""
    phenotypes = disease_phenotypes(kg, r)
    pipeline = {t for s in r["steps"] for f in s["findings"]
                if f.get("status") == "present" for t in f.get("hpo") or []}
    found = {}
    for pid, v in entry["found"].items():
        found[pid] = {**v, "name": kg.vocab.term_name(pid),
                      "in_pipeline": any(pid in kg.vocab.ancestors(c) for c in pipeline),
                      "specific": kg.n_annotated(pid) <= GENERIC * kg.n}
    doctor = [v for v in found.values() if v["step"] > 0]
    n_steps = sum(s["kind"] != "given" for s in r["steps"])
    cols = {
        "dx_phenotypes_total": len(phenotypes),
        "dx_phenotypes_step0": sum(v["step"] == 0 for v in found.values()),
        "dx_phenotypes_doctor": len(doctor),
        "dx_phenotypes_dialogue": len(found),
        "dx_phenotypes_specific": sum(v["specific"] for v in found.values()),
        "dx_phenotypes_missed_by_pipeline": sum(not v["in_pipeline"] for v in found.values()),
        "dx_first_doctor_step": min((v["step"] for v in doctor), default=None),
        "dx_useful_step_fraction": len({v["step"] for v in doctor}) / n_steps if n_steps else None,
        "dx_has_step0": entry["has_step0"],
        "dx_rejected_quote": entry["rejected_quote"],
        "dx_rejected_verify": entry["rejected_verify"],
    }
    return cols, found


# ---------------------------------------------------------------- 2. rescore
def step0_findings(kg, case, referral, cache):
    if case not in cache:
        step = [{"step": 0, "turn": 0, "kind": "given", "action": "referral", "answer": referral}]
        f = k.extract_findings(step).get(0, [])
        cache[case] = {"findings": f, "links": k.link_terms(kg, {0: f}) if f else {}}
        save_json(STEP0_CACHE, cache)
    return cache[case]


def add_finding(findings, links, step, term, status, hpo, tag):
    f = {"term": term, "status": status, "alt": [tag], "hpo": None, "from_step": step}
    findings.setdefault(step, []).append(f)
    links[k._fkey(f)] = ({"id": hpo[0], "same": list(hpo[1:])} if hpo
                         else {"id": None, "same": [], "reason": "unlinked"})


def rescore_case(kg, r, refs, step0, found, doctor_links):
    """kg_eval row recomputed with step 0 and the phenotypes recovered by the check."""
    findings, links = {}, {}
    for s in r["steps"]:
        for i, n in enumerate(s["findings"]):
            add_finding(findings, links, s["step"], n["term"], n["status"], n.get("hpo"), f"o{s['step']}.{i}")
    ref = referral_of(r, refs)
    if ref:
        s0 = step0_findings(kg, r["case"], ref, step0)
        for f in s0["findings"]:
            g, _ = k.resolve(kg, s0["links"].get(k._fkey(f)), f["status"])
            add_finding(findings, links, 0, f["term"], f["status"], sorted(g) if g else None, "step0")
    added = 0
    for pid, v in found.items():
        if not v["in_pipeline"]:
            add_finding(findings, links, int(v["step"]), v["name"], "present", [pid], f"dx{pid}")
            added += 1

    steps = ([{"step": 0, "turn": 0, "kind": "given", "action": "referral", "answer": ref}] if ref else [])
    steps += [{x: s.get(x) for x in ("step", "turn", "kind", "action", "answer", "image_reading")}
              for s in r["steps"] if s["kind"] != "given"]
    k.parse_transcript = lambda transcript, referral="": steps        # reuse kg_eval.evaluate_case
    result = {"case": r["case"], "models": {"doctor": r.get("doctor_model")},
              "true_diagnosis": r["true_diagnosis"], "doctor_diagnosis": r["doctor_diagnosis"],
              "verdict": r["verdict"], "transcript": [r["case"], "rescore"]}
    match = r["kg_match"]
    cache = {"findings": {str(s): v for s, v in findings.items()}, f"links_{kg.key}": links,
             f"dx_{kg.key}": {"exact": r["kg_diagnosis_ids"] if match == "exact" else [],
                              "broader": r["kg_diagnosis_ids"] if match == "broader" else [],
                              "note": r.get("kg_diagnosis_note", "")}}
    if doctor_links is not None:
        cache[f"doctor_dx4_{kg.key}"] = doctor_links
    row = k.evaluate_case(kg, result, {k._key(result): cache})
    row.update(case_key=r["case_key"], kg_diagnosis_note=r.get("kg_diagnosis_note", ""),
               dx_phenotypes_added=added, referral=ref, dx_phenotype_check=found)
    return row


def rescore(model, check_cache=None, kg_caches=None, llm=LLM):
    configure_llm(llm)
    kg = k.load_graph(k.HPOVocab(), "hpo")
    refs, step0 = load_json(REFERRALS, {}), load_json(STEP0_CACHE, {})
    checks = {key: legacy_check(e) for key, e in load_json(check_cache or f"dx_check_{model}.json", {}).items()}
    old = {}
    for f in kg_caches or [f"kg_cache_{model}.json", "kg_cache.json"]:
        old.update(load_json(f, {}))
    progress = Path(f"kg_eval_{model}_final_progress.jsonl")
    done = read_jsonl(progress)
    keys = {x["row"]["case_key"] for x in done}
    cases = kg_cases(model)
    print(f"already rescored: {len(keys)}/{len(cases)}")

    for i, r in enumerate(cases, 1):
        if r["case_key"] in keys:
            continue
        entry = checks.get(r["case_key"], {"found": {}, "rejected_quote": 0, "rejected_verify": 0,
                                           "has_step0": False})
        dx_cols, found = dx_columns(kg, r, entry)
        try:
            row = rescore_case(kg, r, refs, step0, found,
                               (old.get(r["case_key"]) or {}).get(f"doctor_dx4_{kg.key}"))
        except k.MappingFailed as e:
            print(f"{i}/{len(cases)} {r['case']}: LLM error ({e}), retried next run")
            continue
        new, before = trajectory(kg, row), trajectory(kg, r)
        row.update(dx_cols)
        row.update({m: v for m, v in new["summary"].items() if m.startswith(("eig_true", "info_true"))})
        orig = {f"{m}_orig": r.get(m) for m in ORIG_COLUMNS}
        orig.update({f"{m}_orig": v for m, v in before["summary"].items() if m.startswith(("eig_true", "info_true"))})
        row.update(orig)
        done.append({"row": row})
        with open(progress, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"row": row}, ensure_ascii=False) + "\n")
        write_final_csv(model, [x["row"] for x in done])
        print(f"{i}/{len(cases)} {r['case']}: rank {r['final_true_rank']} -> {row['final_true_rank']} | "
              f"{r['error_type']} -> {row['error_type']} | added {row['dx_phenotypes_added']} | "
              f"eig_true_ratio {orig['eig_true_ratio_orig']} -> {row['eig_true_ratio']}", flush=True)

    rows = [x["row"] for x in done]
    save_json(f"kg_eval_{model}_final.json", {"summary": k.summarise(rows), "cases": rows}, indent=2)
    df = write_final_csv(model, rows)
    print(f"saved kg_eval_{model}_final.json/.csv ({len(rows)} cases)")
    print(pd.crosstab(df["error_type_orig"], df["error_type"], margins=True))


def write_final_csv(model, rows):
    extra = sorted({c for r in rows for c in r if c.startswith(("dx_", "eig_true", "info_true"))
                    or c.endswith("_orig")} - {"dx_phenotype_check", "dx_phenotypes_added"})
    df = pd.DataFrame([k._csv_row(r) for r in rows])[k.CSV_FIELDS + ["dx_phenotypes_added"] + extra]
    df.to_csv(f"kg_eval_{model}_final.csv", index=False)
    return df


# ------------------------------------------------------------- 3. robustness
def robustness(model, exact_only=True, with_eig=True, n_cases=None):
    from scipy.stats import mannwhitneyu, spearmanr
    kg = k.load_graph(k.HPOVocab(), "hpo")
    cases = final_cases(model)
    if exact_only:
        cases = [r for r in cases if r.get("kg_match") == "exact"]
    cases = cases[:n_cases] if n_cases else cases
    metrics = [m for m in PAPER_METRICS if m != "mapped_fraction" and (with_eig or m != "eig_true_ratio")]
    print(f"{len(cases)} cases")

    rows, t0 = [], time.time()
    for name, change in VARIANTS.items():
        for p, v in {**BASE_PARAMS, **change}.items():
            setattr(k, p, v)
        for r in cases:
            t = trajectory(kg, r, with_eig)
            if t:
                rows.append({"variant": name, "case_key": r["case_key"], "verdict": r["verdict"],
                             **{m: t["summary"][m] for m in metrics}})
        print(f"{name:20s} done ({time.time() - t0:.0f} s)", flush=True)
    for p, v in BASE_PARAMS.items():
        setattr(k, p, v)
    df = pd.DataFrame(rows)
    df.to_csv(f"robustness_{model}.csv", index=False)

    base = df[df.variant == "baseline"].set_index("case_key")
    summary = []
    for name in VARIANTS:
        d = df[df.variant == name].set_index("case_key")
        for m in metrics:
            g = {v: d[d.verdict == v][m].dropna() for v in ("correct", "partially correct", "incorrect")}
            c, i = g["correct"], g["incorrect"]
            summary.append({
                "variant": name, "metric": m,
                "median_correct": c.median(), "median_partial": g["partially correct"].median(),
                "median_incorrect": i.median(),
                "auc_correct_vs_incorrect": mannwhitneyu(c, i).statistic / (len(c) * len(i)) if len(c) and len(i) else None,
                "spearman_vs_baseline": spearmanr(base[m], d[m].reindex(base.index), nan_policy="omit")[0],
            })
    summary = pd.DataFrame(summary)
    summary.to_csv(f"robustness_{model}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for m in metrics:
        print(f"\n=== {m} ===")
        print(summary[summary.metric == m].drop(columns="metric").round(3).to_string(index=False))


# ------------------------------------------------------------------- 4. plot
def plot(model, case, out=None):
    import matplotlib.pyplot as plt
    kg = k.load_graph(k.HPOVocab(), "hpo")
    r = next(x for x in final_cases(model) if x["case"] == case)
    t = trajectory(kg, r)
    st = t["steps"]
    arr = lambda key: np.array([np.nan if d[key] is None else d[key] for d in st], float)
    run_mean = lambda a: np.array([np.nanmean(a[:i + 1]) if np.any(~np.isnan(a[:i + 1])) else np.nan
                                   for i in range(len(a))])
    x, rank, ig, rho = np.array([d["step"] for d in st]), arr("rank"), arr("ig"), arr("rho")
    mapped, test = np.array([d["mapped"] for d in st]), np.array([d["kind"] == "test" for d in st])

    blue, grey, ink, ink2, muted, bg = "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#d9d8d3", "axes.labelcolor": ink2,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.facecolor": bg, "figure.facecolor": bg})
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(hspace=0.62, wspace=0.18, top=0.83, bottom=0.07, left=0.06, right=0.98)

    def panel(ax, title, value, note):
        ax.text(0, 1.30, title, transform=ax.transAxes, fontsize=10.5, color=ink2, va="bottom")
        ax.text(0, 1.05, value, transform=ax.transAxes, fontsize=17, color=ink, fontweight="bold", va="bottom")
        ax.text(1, 1.07, note, transform=ax.transAxes, fontsize=8.5, color=muted, va="bottom", ha="right")
        ax.grid(axis="y", color="#ecebe7", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for xi in x[test]:
            ax.axvspan(xi - 0.5, xi + 0.5, color="#f1f0ec", lw=0, zorder=0)

    fmt = lambda v, f="{:.2f}": "n/a" if v is None or np.isnan(v) else f.format(v)

    ax = axs[0, 0]
    ax.plot(x, rank, color=grey, lw=1.2)
    ax.plot(x, np.minimum.accumulate(rank), color=blue, lw=2)
    ax.set_yscale("log")
    ax.invert_yaxis()
    panel(ax, "Best rank  $r^{\\min}$", fmt(np.nanmin(rank), "{:g}"), "blue: best so far · grey: current · log scale, top = 1")

    ax = axs[0, 1]
    ax.axhline(0, color=muted, lw=1, ls=(0, (3, 3)))
    ax.plot(x, arr("log_odds"), color=blue, lw=2)
    panel(ax, "Final log-odds  $\\lambda_S$", fmt(arr("log_odds")[-1], "{:+.2f} bit"), "> 0: truth leads its best competitor")

    ax = axs[1, 0]
    info = arr("info_true")
    ax.axhline(0, color=muted, lw=1, ls=(0, (3, 3)))
    ax.fill_between(x, 0, info, color=blue, alpha=0.12, lw=0, step="post")
    ax.step(x, info, where="post", color=blue, lw=2)
    panel(ax, "Information towards the truth  $I_V$", fmt(info[-1], "{:+.2f} bit"),
          "cumulative $\\log_2\\pi_s(V)-\\log_2\\pi_0(V)$")

    ax = axs[1, 1]
    ok = ~np.isnan(rho)
    ax.scatter(x[ok], rho[ok], s=26, color=grey, edgecolor=bg, lw=1.5, zorder=3)
    ax.plot(x, run_mean(rho), color=blue, lw=2)
    ax.set_ylim(-0.03, 1.05)
    panel(ax, "Truth-directed EIG ratio  $\\bar\\rho$", fmt(run_mean(rho)[-1]), "dots: $\\rho_s$ per judged step · line: running mean")

    ax = axs[2, 0]
    ok = ~np.isnan(ig)
    ax.bar(x[ok], ig[ok], width=0.6, color=grey, lw=0)
    ax.axhline(0, color=muted, lw=0.8)
    ax.plot(x, run_mean(ig), color=blue, lw=2)
    panel(ax, "Mean information gain  $\\overline{\\mathrm{IG}}$", fmt(run_mean(ig)[-1], "{:.2f} bit"),
          "bars: IG$_s$ per informative step · line: running mean")
    ax.set_xlabel("doctor step $s$")

    ax = axs[2, 1]
    frac = np.cumsum(mapped) / np.arange(1, len(st) + 1)
    ax.plot(x, frac, color=blue, lw=2)
    ax.plot(x[mapped], np.full(mapped.sum(), -0.04), "|", color=ink2, ms=7, mew=1.2)
    ax.set_ylim(-0.08, 1.05)
    panel(ax, "Mapped fraction  $|\\mathcal{I}|/S$", fmt(frac[-1]), "ticks: steps with new findings")
    ax.set_xlabel("doctor step $s$")

    fig.text(0.06, 0.968, f"{case.split('/')[-1]} — true: {r['true_diagnosis'][:70]}",
             fontsize=13, color=ink, fontweight="bold")
    fig.text(0.06, 0.94, f"doctor: {str(r['doctor_diagnosis']).replace('*', '')[:90]} ({r['verdict']})"
             "   ·   shaded columns = tests", fontsize=9.5, color=ink2)
    out = out or f"trajectory_{case.split('/')[-1].replace(' ', '_')}.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    return fig


# --------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--model", required=True)
    p.add_argument("--cache")
    p.add_argument("--llm", default=LLM)
    p = sub.add_parser("rescore")
    p.add_argument("--model", required=True)
    p.add_argument("--check-cache")
    p.add_argument("--kg-caches", nargs="*")
    p.add_argument("--llm", default=LLM)
    p = sub.add_parser("robustness")
    p.add_argument("--model", required=True)
    p.add_argument("--all-matches", action="store_true")
    p.add_argument("--no-eig", action="store_true")
    p.add_argument("--n-cases", type=int)
    p = sub.add_parser("plot")
    p.add_argument("--model", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--out")
    a = ap.parse_args()

    if a.cmd == "check":
        check(a.model, a.cache, a.llm)
    elif a.cmd == "rescore":
        rescore(a.model, a.check_cache, a.kg_caches, a.llm)
    elif a.cmd == "robustness":
        robustness(a.model, not a.all_matches, not a.no_eig, a.n_cases)
    else:
        plot(a.model, a.case, a.out)


if __name__ == "__main__":
    main()
