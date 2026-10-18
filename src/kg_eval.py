"""
Post-hoc evaluation of the doctor's questions against a disease-phenotype knowledge graph.

Default graph: the OMIM + Orphanet disease annotations of the Human Phenotype Ontology, as
shipped with pyhpo (GRAPH = "hpo"). PrimeKG (Chandak, Huang & Zitnik, Scientific Data 2023) is
still available with GRAPH = "primekg". Only structured disease-phenotype annotations are used:
no LLM writes any disease profile. HPO terms also serve as the vocabulary of findings.

Reads the results.json produced by the arena (a list of sim.run() results); the simulation is
not touched. Meant to run after kg_coverage.py: its diagnosis links (kg_coverage.json) are
reused, so exactly the cases it marked "covered" are scored, and the others cost no LLM call.

Pipeline, per case:
  1. The true diagnosis is linked to graph nodes (from kg_coverage.json, else by the LLM).
     Cases not covered (or with fewer than MIN_DX_PHENOTYPES phenotypes) stop here.
  2. The transcript is split into steps (ASK + answer, TEST + result + image reading); the
     referral line, if present, is step 0 ("given", not credited to the doctor).
  3. An LLM turns each step into findings of the CURRENT illness, present or absent, linked to
     HPO terms chosen among retrieved candidates (identifiers are never invented). Equivalent
     terms are grouped; a PRESENT finding coded with a term no disease uses falls back to its
     nearest used ancestors.
  4. The doctor's own diagnosis is linked to the graph the same way (for metrics A2 and B2).
  5. A naive-Bayes model over all diseases with phenotypes is recomputed at every step from the
     evidence gathered so far (repeats ignored, later status wins, hierarchy-implied findings
     not counted twice).

Metrics (see kg_metrics.tex for the formulas):
  per step   info_gain_bits, eig_asked_bits, eig_best_bits, best_question, saturated,
             true_rank, true_posterior, log_odds_true
  per case   efficiency: mapped/unmapped/redundant/low-value fractions, mean_eig_ratio,
               info_regret_bits (C1: sum over judged steps of best EIG - asked EIG)
             reachability: final/min rank, percentile, band, posterior,
               final/max_log_odds_true (B3: log2 P(true) / P(best other disease)),
               steps_to_top1/10/50 (C2: first step with the true diagnosis in the top-K)
             doctor's diagnosis: doctor_dx_rank, diagnostic_regret_bits (A2: log2 P(true) -
               log2 P(doctor's); > 0 = the data favoured the true diagnosis),
               n_discriminating (B2: observed findings annotated in only one of the two)
             error_type: correct / synthesis_error (rank <= REACH_K) / gathering_error
               (rank > UNREACH_K) / undetermined / kg_not_covered / mapping_failed

All of this is "according to the graph". Report coverage, match type and the calibration check
(ranks of correct vs incorrect cases, in the summary) together with any result.

Usage:
    python kg_coverage.py results.json --graphs hpo --model genai:gemini-3.5-flash-lite
    python kg_eval.py results.json --model genai:gemini-3.5-flash-lite
    # from a notebook:
    import kg_eval
    summary = kg_eval.evaluate("results.json", model="genai:gemini-3.5-flash-lite")

Outputs: kg_eval.json (per case, step by step), kg_eval.csv (one row per case), kg_cache.json
(per-case LLM mappings, reused on later runs).
Requires: pip install pyhpo numpy   (pandas only for PrimeKG; clinical_sim.py importable)
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import clinical_sim as sim

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
MODEL = None                  # None = sim.EXAMINER_MODEL; any "provider:model" accepted by sim.chat
KG_PATH = "kg.csv"            # PrimeKG edge list (Harvard Dataverse)
KG_CACHE = "primekg_phenotypes.json"
GRAPH = "hpo"                 # "hpo": OMIM + Orphanet annotations shipped with pyhpo (default);
                              # "primekg": PrimeKG (needs kg.csv or primekg_phenotypes.json)
COVERAGE_PATH = "kg_coverage.json"   # output of kg_coverage.py: its diagnosis links are reused,
                              # so the cases scored are exactly those it marked "covered"
PROMPT_VERSION = "v2"         # part of the cache key: change it when the findings prompt changes

P_HIT = 0.6                   # P(finding present | disease lists it, or a more specific form of it).
                              # Annotation lists mix obligatory and occasional features and PrimeKG
                              # has no frequencies: 0.8 made the absence of an occasional feature
                              # (e.g. diarrhoea in ectopic ACTH) cost a factor 4.5.
P_BACKGROUND = 0.1            # P(finding present | disease does not list it): the graph is incomplete
P_EXCLUDED = 0.02             # P(finding present | disease explicitly listed as NOT having it)
P_PARTIAL = 0.3               # P(finding present | disease lists only a more GENERAL form of it,
                              # e.g. "Glomerulopathy" for an observed "Crescentic glomerulonephritis").
                              # Annotations are often coarser than the findings: without this the
                              # true disease got no credit at all. Only ancestors found in at most
                              # PARTIAL_MAX_PREVALENCE of the diseases count (not headings).
PARTIAL_MAX_PREVALENCE = 0.05
ABSENT_WEIGHT = 0.3           # weight of an ABSENT finding in the update (tempered likelihood).
                              # Annotation lists mix obligatory and occasional features, so a denied
                              # symptom is weak evidence against a disease that lists it; at full
                              # weight, diseases with many annotations fell below the unrelated ones
                              # (in GLM runs, 8 correctly diagnosed cases had the truth ranked >1000).
DAMPING = 0.0                 # evidence weight of each finding = (findings in its step) ** -DAMPING.
                              # 0 = naive Bayes; 0.5 softens panels of correlated results
                              # (e.g. acidosis + anion gap + hypokalaemia from one test)

MIN_DX_PHENOTYPES = 5         # a diagnosis node needs this many phenotype relations to be scored
REACH_K = 10                  # final rank <= REACH_K  -> the data gathered was enough
UNREACH_K = 50                # final rank >  UNREACH_K -> it was not; in between: undetermined
LOW_EIG_RATIO = 0.2           # a mapped step is "low value" if its EIG is below this fraction of
                              # the best EIG available at that moment (relative: under a flat
                              # posterior every single question is worth little in absolute bits) ...
SATURATED_BITS = 1.0          # ... unless the posterior entropy is already below this: nothing
                              # left to learn, so the step is not judged
TOP_DISEASES_FOR_BEST = 30    # the best available question is searched among these diseases' findings
MAX_TERM_PREVALENCE = 0.05    # best-question candidates: terms found in at most 5% of diseases
                              # (excludes headings such as "Abnormality of the eye")
BACKOFF_MAX_PREVALENCE = 0.02 # fallback ancestor of an unused term must be at least this specific

STEPS_PER_CALL = 25           # transcript steps sent to the LLM per extraction call
LINKS_PER_CALL = 30           # findings linked per call
TERM_CANDIDATES = 6           # candidates retrieved per name (finding name and each alternative)
MAX_LINK_CANDIDATES = 15
DISEASE_CANDIDATES = 25
EIG_BATCH = 256               # terms per expected-gain batch (memory)
DISC_TAU = 0.4                # a finding discriminates two diagnoses if |p_v - p_m| >= DISC_TAU
                              # (with the defaults: annotated in exactly one of the two)
CONVERGENCE_KS = (1, 10, 50)  # steps_to_topK is reported for these K
VERBOSE_UNCOVERED = False     # also print a line for each case not covered by the graph

# result keys that may hold the presenting complaint (the referral line); first non-empty wins
REFERRAL_KEYS = ("referral", "presenting_complaint", "chief_complaint", "opening_line", "opening")

STOPWORDS = {"of", "the", "and", "in", "on", "with", "to", "a", "an", "or", "for", "by", "at",
             "type", "disease", "syndrome", "disorder"}


def _tokens(text, drop_stop=True):
    toks = [t[:-1] if len(t) > 4 and t.endswith("s") else t
            for t in re.findall(r"[a-z0-9]+", str(text).lower())]
    return [t for t in toks if not (drop_stop and t in STOPWORDS)]


# --------------------------------------------------------------------------- #
# HPO vocabulary (terms, synonyms, hierarchy only; no diseases)
# --------------------------------------------------------------------------- #
class HPOVocab:
    def __init__(self):
        from pyhpo import Ontology
        print("[kg] loading the HPO vocabulary (about 15 s)...", flush=True)
        Ontology()
        self.ont = Ontology
        root = Ontology.get_hpo_object("HP:0000118")       # phenotypic abnormality subtree
        seen, stack = set(), [root]
        while stack:
            t = stack.pop()
            if t.id in seen:
                continue
            seen.add(t.id)
            stack.extend(t.children)
        self.phenotype_ids = seen
        self.term_labels, self.inv, self._anc = {}, {}, {}
        for tid in seen:
            t = Ontology.get_hpo_object(tid)
            labels = [t.name] + list(t.synonym or [])
            self.term_labels[tid] = labels
            for lab in labels:
                for tok in set(_tokens(lab)):
                    self.inv.setdefault(tok, set()).add(tid)

    def term(self, tid):
        if not tid:
            return None
        try:
            t = self.ont.get_hpo_object(str(tid))
        except (KeyError, ValueError, RuntimeError, TypeError, NameError):
            return None
        return t if t.id in self.phenotype_ids else None

    def term_name(self, tid):
        t = self.term(tid)
        return t.name if t else str(tid)

    def ancestors(self, tid):
        """The term and all its ancestors (phenotype subtree only); empty if unknown."""
        if tid not in self._anc:
            t = self.term(tid)
            self._anc[tid] = (({tid} | {p.id for p in t.all_parents}) & self.phenotype_ids
                              if t else set())
        return self._anc[tid]

    def parents(self, tid):
        t = self.term(tid)
        return [p.id for p in t.parents if p.id in self.phenotype_ids] if t else []

    def search_terms(self, phrase, k=TERM_CANDIDATES):
        q = _tokens(phrase)
        if not q:
            return []
        qs, qnorm = set(q), " ".join(_tokens(phrase, drop_stop=False))
        pool = set().union(*(self.inv.get(t, set()) for t in qs))
        scored = []
        for tid in pool:
            best = 0.0
            for lab in self.term_labels[tid]:
                if " ".join(_tokens(lab, drop_stop=False)) == qnorm:
                    best = 3.0
                    break
                ls = set(_tokens(lab))
                if ls:
                    inter = len(qs & ls)
                    best = max(best, inter / len(qs) + 0.5 * inter / len(ls))
            scored.append((best, tid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [tid for _, tid in scored[:k]]


# --------------------------------------------------------------------------- #
# PrimeKG
# --------------------------------------------------------------------------- #
def _read_primekg(kg_path, cache_path):
    """{disease id: {"name", "pos": [HP ids], "neg": [HP ids]}} for EVERY disease node.

    Read once from kg.csv (only disease rows are kept) and cached as compact JSON.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    import pandas as pd
    kg_path = Path(kg_path)
    if not kg_path.exists():
        raise FileNotFoundError(
            f"{kg_path} not found. Download PrimeKG's kg.csv from Harvard Dataverse "
            "(https://dataverse.harvard.edu/api/access/datafile/6180620) and pass its path.")
    print(f"[kg] reading {kg_path} once (a few minutes)...", flush=True)
    cols = ["relation", "x_id", "x_type", "x_name", "x_source", "y_id", "y_type", "y_source"]
    out = {}
    for chunk in pd.read_csv(kg_path, usecols=cols, dtype=str, chunksize=1_000_000,
                             low_memory=False):
        dis = chunk[chunk["x_type"] == "disease"]
        for src, xid, name in dis[["x_source", "x_id", "x_name"]].drop_duplicates().itertuples(
                index=False):
            out.setdefault(f"{src}:{xid}", {"name": name, "pos": [], "neg": []})
        ph = dis[dis["relation"].isin(["disease_phenotype_positive",
                                       "disease_phenotype_negative"])
                 & (dis["y_type"] == "effect/phenotype")]
        for rel, src, xid, ysrc, yid in ph[["relation", "x_source", "x_id", "y_source",
                                            "y_id"]].itertuples(index=False):
            try:
                hp = f"HP:{int(yid):07d}" if ysrc == "HPO" else None
            except ValueError:
                hp = None
            if hp:
                out[f"{src}:{xid}"]["pos" if rel.endswith("positive") else "neg"].append(hp)
    for d in out.values():
        d["pos"], d["neg"] = sorted(set(d["pos"])), sorted(set(d["neg"]))
    cache_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    n_ph = sum(1 for d in out.values() if d["pos"])
    print(f"[kg] {len(out)} disease nodes, {n_ph} with phenotype relations -> {cache_path}")
    return out


def _read_hpo():
    """Same format as _read_primekg, from the OMIM and Orphanet annotations shipped with pyhpo."""
    from pyhpo import Ontology

    def ids(xs):
        return sorted({f"HP:{int(i):07d}" for i in (xs or [])})

    out = {}
    for src, diseases in (("OMIM", Ontology.omim_diseases), ("ORPHA", Ontology.orpha_diseases)):
        for d in diseases:
            out[f"{src}:{d.id}"] = {"name": d.name, "pos": ids(d.hpo),
                                    "neg": ids(getattr(d, "negative_hpo", None))}
    return out


def load_graph(vocab, graph=None, kg_path=KG_PATH, kg_cache=KG_CACHE):
    graph = (graph or GRAPH).lower()
    if graph == "hpo":
        return DiseaseGraph(vocab, _read_hpo(), "hpo", "HPO (OMIM + Orphanet)")
    if graph == "primekg":
        return DiseaseGraph(vocab, _read_primekg(kg_path, kg_cache), "primekg", "PrimeKG")
    raise ValueError(f"unknown graph {graph!r}: use 'hpo' or 'primekg'")


class DiseaseGraph:
    def __init__(self, vocab, data, key="hpo", label="HPO (OMIM + Orphanet)"):
        self.vocab = vocab
        self.key, self.label = key, label

        # every node can be named as the diagnosis; only nodes with phenotypes can be scored
        self.all_ids = list(data)
        self.all_names = [data[i]["name"] for i in self.all_ids]
        self.all_index = {x: k for k, x in enumerate(self.all_ids)}
        self.ids = [i for i in self.all_ids if data[i]["pos"]]
        self.names = [data[i]["name"] for i in self.ids]
        self.col = {x: c for c, x in enumerate(self.ids)}
        self.n = len(self.ids)

        # true-path rule: a disease with a specific phenotype also has its more general forms
        self.direct = [[t for t in data[i]["pos"] if vocab.term(t)] for i in self.ids]
        member, excluded = {}, {}
        for c, terms in enumerate(self.direct):
            for t in terms:
                for a in vocab.ancestors(t):
                    member.setdefault(a, set()).add(c)
        for c, i in enumerate(self.ids):
            for t in data[i]["neg"]:
                excluded.setdefault(t, set()).add(c)
        self._member_sets, self._excluded_sets = member, excluded
        self._member, self._excluded, self._backoff, self._partial = {}, {}, {}, {}

        self.dis_inv = {}
        for k, name in enumerate(self.all_names):
            for tok in set(_tokens(name)):
                self.dis_inv.setdefault(tok, set()).add(k)
        print(f"[kg] {label}: {len(self.all_ids)} disease nodes, {self.n} with phenotype "
              f"relations, {len(member)} phenotype terms in use")

    # ---- names and counts ------------------------------------------------- #
    def term_name(self, tid):
        return self.vocab.term_name(tid)

    def group_name(self, group):
        return " / ".join(self.term_name(t) for t in sorted(group))

    def n_annotated(self, tid):
        """Diseases that have the term or a more specific form of it."""
        return len(self._member_sets.get(tid, ()))

    def n_phenotypes(self, node_id):
        c = self.col.get(node_id)
        return len(self.direct[c]) if c is not None else 0

    def all_name(self, node_id):
        k = self.all_index.get(node_id)
        return self.all_names[k] if k is not None else node_id

    # ---- likelihoods ------------------------------------------------------ #
    def member(self, tid):
        if tid not in self._member:
            m = np.zeros(self.n, dtype=bool)
            m[list(self._member_sets.get(tid, ()))] = True
            self._member[tid] = m
        return self._member[tid]

    def excluded(self, tid):
        if tid not in self._excluded:
            m = np.zeros(self.n, dtype=bool)
            m[list(self._excluded_sets.get(tid, ()))] = True
            self._excluded[tid] = m
        return self._excluded[tid]

    def partial(self, group):
        """Diseases that list a more general (but specific enough) form of any term of the group."""
        key = frozenset(group)
        if key not in self._partial:
            m = np.zeros(self.n, dtype=bool)
            for t in key:
                for a in self.vocab.ancestors(t):
                    if a != t and 0 < self.n_annotated(a) <= PARTIAL_MAX_PREVALENCE * self.n:
                        m |= self.member(a)
            self._partial[key] = m
        return self._partial[key]

    def p_present(self, group, partial=True):
        """P(finding present | disease) for a group of equivalent terms: the disease has the
        finding if it has ANY of them (P_HIT); otherwise, if it lists a more general form of it,
        P_PARTIAL; it is excluded (P_EXCLUDED) only if ALL of them are excluded."""
        group = list(group)
        has = np.zeros(self.n, dtype=bool)
        exc = np.ones(self.n, dtype=bool)
        for t in group:
            has |= self.member(t)
            exc &= self.excluded(t)
        p = np.where(has, P_HIT, P_BACKGROUND)
        if partial and P_PARTIAL:
            p = np.where(~has & self.partial(group), P_PARTIAL, p)
        return np.where(exc & ~has, P_EXCLUDED, p)

    def prevalence(self, group):
        return float(np.mean(np.logical_or.reduce([self.member(t) for t in group])))

    def backoff(self, tid):
        """Nearest ancestors used by some disease, if specific enough; [] if none."""
        if tid in self._backoff:
            return self._backoff[tid]
        out, frontier, seen = [], [tid], {tid}
        while frontier and not out:
            nxt = []
            for t in frontier:
                for p in self.vocab.parents(t):
                    if p in seen:
                        continue
                    seen.add(p)
                    if self.n_annotated(p):
                        out.append(p)
                    else:
                        nxt.append(p)
            frontier = nxt
        out = [p for p in out if self.n_annotated(p) / self.n <= BACKOFF_MAX_PREVALENCE]
        self._backoff[tid] = sorted(out)
        return self._backoff[tid]

    def disease_terms(self, c):
        return self.direct[c]

    def search_diseases(self, names, k=DISEASE_CANDIDATES):
        """Indices into all_ids (nodes with or without phenotypes)."""
        scores = {}
        for name in names:
            qs = set(_tokens(name)) or set(_tokens(name, drop_stop=False))
            if not qs:
                continue
            pool = set().union(*(self.dis_inv.get(t, set()) for t in qs))
            for k_ in pool:
                ds = set(_tokens(self.all_names[k_])) or {"_"}
                inter = len(qs & ds)
                s = inter / len(qs) + 0.5 * inter / len(ds)
                scores[k_] = max(scores.get(k_, 0.0), s)
        ranked = sorted(scores, key=lambda k_: (-scores[k_], self.all_names[k_]))
        return ranked[:k]


# --------------------------------------------------------------------------- #
# Transcript -> steps
# --------------------------------------------------------------------------- #
def _image_reading(entry):
    """Text of a 'DOCTOR READS (file): ...' line, without the reader's differential."""
    text = entry.split(":", 1)[1] if ":" in entry else ""
    text = re.split(r"\bDIFFERENTIAL IMPACT\s*:", text, maxsplit=1)[0]
    return " ".join(text.split())


def referral_text(result):
    for k in REFERRAL_KEYS:
        v = result.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def parse_transcript(transcript, referral=""):
    """[{"step", "turn", "kind": "given"|"ask"|"test", "action", "answer"}] from a transcript.

    Step 0 ("given") holds the referral line when known. Image readings are appended to the
    answer of the test that produced the image.
    """
    steps, turn, pending = [], 0, None
    if referral:
        steps.append({"step": 0, "turn": 0, "kind": "given", "action": "referral",
                      "answer": referral})
    n = 0
    for entry in transcript:
        if not isinstance(entry, str):
            continue
        if entry.startswith("DOCTOR: ") or entry.startswith("DOCTOR (forced)"):
            action = entry.split(": ", 1)[1] if ": " in entry else ""
            if entry.startswith("DOCTOR: "):
                turn += 1
            up = action.upper()
            if up.startswith("ASK:"):
                pending = ("ask", action.split(":", 1)[1].strip())
            elif up.startswith("TEST:"):
                pending = ("test", action.split(":", 1)[1].strip())
            else:
                pending = None
        elif entry.startswith("PATIENT: ") and pending and pending[0] == "ask":
            n += 1
            steps.append({"step": n, "turn": turn, "kind": "ask",
                          "action": pending[1], "answer": entry[len("PATIENT: "):].strip()})
            pending = None
        elif entry.startswith("RESULT (") and pending and pending[0] == "test":
            prefix = f"RESULT ({pending[1]})"
            rest = entry[len(prefix):] if entry.startswith(prefix) else entry.split(")", 1)[-1]
            rest = re.sub(r"^\s*\[[^\]]*\]", "", rest).lstrip(": ").strip()
            n += 1
            steps.append({"step": n, "turn": turn, "kind": "test",
                          "action": pending[1], "answer": rest})
            pending = None
        elif entry.startswith("DOCTOR READS"):
            reading = _image_reading(entry)
            target = next((s for s in reversed(steps) if s["kind"] == "test"), None)
            if reading and target is not None:
                target["answer"] += f"\nIMAGE READING: {reading}"
                target["image_reading"] = True
    return steps


# --------------------------------------------------------------------------- #
# LLM mapping (findings, term links, diagnosis), validated against the graph
# --------------------------------------------------------------------------- #
FINDINGS_PROMPT = """You convert a clinical encounter into phenotypic findings coded with the Human
Phenotype Ontology (HPO). You receive numbered steps: each is a physician's question with the
patient's answer, or an ordered test with its result (for images, with the physician's reading).
Step 0, if present, is the referral line that opened the encounter.

For each step, list the findings of the CURRENT illness that the step establishes, as HPO-style
term names in English (e.g. "Fever", "Dyspnea", "Abdominal pain", "Hypokalemia", "Anemia").
- status "present" if the step establishes the finding; "absent" if the answer denies it for the
  current illness, or if a test result is normal for it (normal hemoglobin -> "Anemia" absent).
- Laboratory values: "present" only if outside the reference range when one is given, or clearly
  abnormal for an adult. If several dates are given, use the most recent value.
- Image readings: code the findings the reading describes (e.g. "Cerebral calcification"), not
  the reader's guess about their cause.
- NOT findings: past medical history and chronic conditions known before this illness ("I have
  recurrent sinus infections" -> nothing); answers about earlier episodes ("Have you ever had a
  seizure before?" "No, never." -> nothing: it says nothing about the current illness);
  medications, diagnoses, dates, durations, demographics, social context, treatments; vague or
  uncertain answers ("I don't know" -> nothing).
- source "answer" if the answer or result establishes the finding. source "question" only if the
  physician's question states it as an already known fact ("this swelling around your eyelids",
  "the rash on your forehead") and the answer does not contradict it.
- Translate lay language into the clinical term ("I can't catch my breath" -> "Dyspnea"). Keep the
  specific type or site when the step gives it ("Vesicular rash of the forehead", not "Rash").
- In "alt", give up to three other names of the SAME finding, as HPO might word them (e.g.
  "Eosinopenia" -> ["Decreased eosinophil count"]; "Tiredness" -> ["Fatigue", "Asthenia"]).
- Use only what the step states: no inference from other steps, no diagnoses.
- If you know the HPO identifier, give it in "hpo"; otherwise null.
- A step may establish nothing: give an empty list.

Reply with JSON only, no markdown:
{"steps": [{"step": 1, "findings": [{"term": "Fever", "alt": ["Elevated body temperature"],
"hpo": "HP:0001945", "status": "absent", "source": "answer"}]}]}"""

LINK_PROMPT = """You link clinical findings to Human Phenotype Ontology terms. For each item you get
the finding and candidate terms; "used by N" is how many diseases of the knowledge graph have the
term (0 = no disease uses it).
- "id": the candidate with the same clinical meaning as the finding. If none has the same
  meaning, the closest more general candidate that still describes the finding, but never a broad
  heading (e.g. "Abnormality of the eye"). Never a more specific candidate. null if none fits.
- If several candidates have the same meaning, choose as "id" the one used by more diseases.
- "same": the OTHER candidates that mean the same thing as "id" (one finding split across terms,
  e.g. "Fatigue" and "Asthenia", "Poor appetite" and "Anorexia"). Not broader, narrower or merely
  related terms. Usually empty.

Reply with JSON only, no markdown:
{"links": [{"item": 0, "id": "HP:0001945", "same": []}, {"item": 1, "id": null, "same": []}]}"""

DX_NAMES_PROMPT = """Give the names under which this diagnosis would be listed in disease ontologies
(MONDO, OMIM, Orphanet): the full name, common synonyms, abbreviations written out, and the more general disease
it is a form of (e.g. "infective endocarditis" for "fungal endocarditis of the aortic valve").
If the diagnosis names a manifestation and its cause ("seizure from neurocysticercosis",
"Cushing's syndrome due to ectopic ACTH from small-cell lung cancer"), give names for the whole
entity and for the cause. At most ten.
Reply with JSON only, no markdown: {"names": ["..."]}"""

DX_PICK_PROMPT = """Choose, among the candidate entries of a disease knowledge graph, those that
correspond to the diagnosis given.
First decide the principal disease: if one entry names the whole diagnosis (manifestation and
cause together, e.g. "Cushing syndrome due to ectopic ACTH secretion"), that is it; otherwise, if
the diagnosis is a manifestation due to a cause ("seizure from neurocysticercosis"), the cause;
if several unrelated diagnoses are listed, the first.
- "exact": entries that ARE the principal disease (a subtype or a grouped entry of that same
  entity is acceptable).
- "broader": only if no entry is exact, the single most specific entry that is the same disease
  process stated less specifically (cause, site or subtype left unspecified), e.g. "infective
  endocarditis" for "fungal endocarditis of the aortic valve".
NEVER acceptable, in either list: a genetic susceptibility or predisposition to the disease; an
immunodeficiency that predisposes to it; a congenital, neonatal or inherited form when the
diagnosis is acquired (or the reverse); a complication or a mere manifestation; a category of
diseases (e.g. "heart disease"); a related or different disease. Both lists may be empty.
Reply with JSON only, no markdown: {"exact": ["..."], "broader": ["..."], "note": "one short sentence"}"""


DOCTOR_PICK_PROMPT = """Choose, among the candidate entries of a disease knowledge graph, those that
correspond to the diagnosis proposed by a physician. The goal is to locate that diagnosis in the
graph, so be inclusive about the form of the entry:
- "exact": entries that are the same disease as the one the physician names: a synonym, a subtype,
  a numbered or genetic entry ("Narcolepsy 1"), an idiopathic or acquired form, a grouped entry.
  If the physician names a disease and its cause ("encephalitis secondary to ovarian teratoma"),
  the disease named first is the diagnosis.
- "broader": only if no entry is exact, the single most specific entry of which the named disease
  is a form (cause, site or subtype left unspecified).
NOT acceptable: a different disease; a category of diseases (e.g. "heart disease"); a mere
symptom or manifestation when the physician names only a symptom (then both lists are empty).
Reply with JSON only, no markdown: {"exact": ["..."], "broader": ["..."], "note": "one short sentence"}"""


class MappingFailed(Exception):
    """An LLM call returned nothing usable: the case is not scored and not cached."""


def _llm(system, user):
    """One validated-JSON call through clinical_sim's back ends. Raises MappingFailed."""
    model = MODEL or sim.EXAMINER_MODEL
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(2):
        raw = sim.chat(model, messages, temperature=0.0, fatal=False)
        if raw is None:
            break
        try:
            d = sim.parse_json_reply(raw)
            if isinstance(d, dict):
                return d
        except (ValueError, AttributeError, TypeError):
            pass
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": "That was not valid JSON. Reply with JSON only, "
                                                 "in the required form."}]
    raise MappingFailed(system.split("\n", 1)[0][:60])


def extract_findings(steps):
    """{step number: [{"term", "alt", "hpo", "status"}]}; question-sourced findings go to step 0."""
    out = {}
    for start in range(0, len(steps), STEPS_PER_CALL):
        chunk = steps[start:start + STEPS_PER_CALL]

        def block(s):
            if s["kind"] == "given":
                return f"STEP 0 (referral)\nREFERRAL: {s['answer']}"
            q = s["kind"] == "ask"
            return (f"STEP {s['step']} ({'question' if q else 'test'})\n"
                    f"{'Q' if q else 'ORDER'}: {s['action']}\n"
                    f"{'A' if q else 'RESULT'}: {s['answer']}")

        d = _llm(FINDINGS_PROMPT, "\n\n".join(block(s) for s in chunk))
        if not isinstance(d.get("steps"), list):
            raise MappingFailed("findings: no 'steps' list")
        valid = {s["step"] for s in chunk}
        for item in d["steps"]:
            try:
                n = int((item or {}).get("step"))
            except (TypeError, ValueError):
                continue
            if n not in valid:
                continue
            for f in item.get("findings") or []:
                if not isinstance(f, dict):
                    continue
                term = str(f.get("term") or "").strip()
                status = str(f.get("status") or "").strip().lower()
                if not term or status not in ("present", "absent"):
                    continue
                alt = [str(a).strip() for a in (f.get("alt") or []) if str(a).strip()][:3]
                source = str(f.get("source") or "answer").strip().lower()
                target = 0 if (source == "question" and n != 0) else n
                out.setdefault(target, []).append(
                    {"term": term, "alt": alt, "hpo": f.get("hpo"), "status": status,
                     "from_step": n})
    return out


def _fkey(f):
    return f"{f['term']}|{f.get('hpo')}|{'|'.join(f.get('alt') or [])}"


def link_terms(kg, findings):
    """{finding key: {"id": HP id or None, "same": [HP ids]}}. Only retrieved candidates."""
    vocab = kg.vocab
    todo = {}
    for fs in findings.values():
        for f in fs:
            todo.setdefault(_fkey(f), f)

    links, items = {}, []
    for key, f in todo.items():
        cands = []
        guess = vocab.term(str(f.get("hpo") or "").strip())
        if guess is not None:                         # canonical id ("HP:001934" -> "HP:0001934")
            cands.append(guess.id)
        for name in [f["term"]] + list(f.get("alt") or []):
            for c in vocab.search_terms(name):
                if c not in cands:
                    cands.append(c)
        cands = cands[:MAX_LINK_CANDIDATES]
        if not cands:
            links[key] = {"id": None, "same": [], "reason": "no_candidate"}
        elif len(cands) == 1 and any(lab.lower() == f["term"].lower()
                                     for lab in vocab.term_labels[cands[0]]):
            links[key] = {"id": cands[0], "same": [], "reason": "exact_only_candidate"}
        else:
            items.append((key, f, cands))

    for start in range(0, len(items), LINKS_PER_CALL):
        chunk = items[start:start + LINKS_PER_CALL]
        text = "\n\n".join(
            f"ITEM {i}: finding \"{f['term']}\""
            + (f" (also called: {', '.join(f['alt'])})" if f.get("alt") else "")
            + "\ncandidates:\n"
            + "\n".join(f"  {c} {vocab.term_name(c)} [used by {kg.n_annotated(c)}]"
                        + (f" (also: {', '.join(vocab.term_labels.get(c, [])[1:4])})"
                           if len(vocab.term_labels.get(c, [])) > 1 else "") for c in cands)
            for i, (_, f, cands) in enumerate(chunk))
        d = _llm(LINK_PROMPT, text)
        if not isinstance(d.get("links"), list):
            raise MappingFailed("links: no 'links' list")
        for link in d["links"]:
            try:
                i = int((link or {}).get("item"))
            except (TypeError, ValueError):
                continue
            if not 0 <= i < len(chunk):
                continue
            key, _, cands = chunk[i]
            chosen = link.get("id") if link.get("id") in cands else None
            same = [x for x in (link.get("same") or []) if x in cands and x != chosen]
            links[key] = {"id": chosen, "same": same if chosen else [],
                          "reason": "llm" if chosen else "llm_null"}
        for key, _, _ in chunk:
            links.setdefault(key, {"id": None, "same": [], "reason": "llm_missing"})
    return links


def diagnosis_names(diagnosis, split_commas=True):
    sep = r"[;,]" if split_commas else r";"
    names = [diagnosis] + [p.strip() for p in re.split(sep, diagnosis) if p.strip()]
    d = _llm(DX_NAMES_PROMPT, f"DIAGNOSIS: {diagnosis}")
    return list(dict.fromkeys(names + [str(n) for n in (d.get("names") or []) if n]))


def link_diagnosis(kg, diagnosis, names, label="ESTABLISHED DIAGNOSIS", prompt=None,
                   extra_ids=()):
    """{"exact": [ids], "broader": [id], "note"}; both empty = not in the graph.
    extra_ids are always offered as candidates (e.g. the true diagnosis' nodes)."""
    cands = kg.search_diseases(names)
    cands += [kg.all_index[x] for x in extra_ids
              if x in kg.all_index and kg.all_index[x] not in cands]
    if not cands:
        return {"exact": [], "broader": [], "note": "no candidate entry found"}
    text = (f"{label}: {diagnosis}\n\nCANDIDATES:\n"
            + "\n".join(f"  {kg.all_ids[k]} {kg.all_names[k]}" for k in cands))
    d = _llm(prompt or DX_PICK_PROMPT, text)
    exact = _resolve_picks(kg, d.get("exact"), cands)
    broader = [] if exact else _resolve_picks(kg, d.get("broader"), cands)[:1]
    return {"exact": exact, "broader": broader, "note": str(d.get("note") or "")}


def _resolve_picks(kg, items, cands):
    """Candidate ids from the model's reply, which may give the id, "id name", or the name."""
    if isinstance(items, (str, dict)):
        items = [items]
    by_id = {kg.all_ids[k]: kg.all_ids[k] for k in cands}
    by_id_norm = {re.sub(r"\s+", "", x).upper(): x for x in by_id}
    by_name = {kg.all_names[k].strip().lower(): kg.all_ids[k] for k in cands}
    out = []
    for item in items or []:
        if isinstance(item, dict):                    # {"id": ..., "name": ...}
            item = item.get("id") or item.get("name") or ""
        s = str(item).strip().strip("\"'")
        hit = by_id.get(s) or by_id_norm.get(re.sub(r"\s+", "", s).upper())
        if not hit:                                   # "ORPHA:397 Giant cell arteritis"
            m = re.match(r"\s*([A-Za-z_]+\s*:\s*[\w.]+)", s)
            if m:
                hit = by_id_norm.get(re.sub(r"\s+", "", m.group(1)).upper())
        if not hit:                                   # the entry name
            name = re.sub(r"^\s*[A-Za-z_]+\s*:\s*[\w.]+\s*", "", s).strip().lower()
            hit = by_name.get(s.lower()) or by_name.get(name)
        if hit and hit not in out:
            out.append(hit)
    return out


# --------------------------------------------------------------------------- #
# Findings -> evidence
# --------------------------------------------------------------------------- #
def resolve(kg, link, status="present"):
    """(group of HP ids, how) for one linked finding; (None, reason) if it carries nothing.

    A term no disease uses falls back to its nearest used ancestors only for a PRESENT finding:
    "eye pain on movement" implies "eye pain", but its absence does not imply no eye pain."""
    if not link or not link.get("id"):
        return None, (link or {}).get("reason", "unlinked")
    terms = [t for t in [link["id"]] + list(link.get("same") or []) if kg.vocab.term(t)]
    if not terms:
        return None, "not_a_phenotype_term"
    used = [t for t in terms if kg.n_annotated(t)]
    if used:
        return frozenset(used), "linked"
    back = sorted({b for t in terms for b in kg.backoff(t)}) if status == "present" else []
    if back:
        return frozenset(back), "backoff"
    return None, "term_unused_in_graph"


def _implied(kg, g, others, status):
    """True if finding group g is implied by another group with the same status:
    present: every term of g is a strict ancestor of a term of some other present group;
    absent:  every term of g has a strict ancestor in some other absent group."""
    anc = kg.vocab.ancestors
    for h in others:
        if h == g:
            continue
        if status == "present":
            if all(any(t != x and t in anc(x) for x in h) for t in g):
                return True
        elif all(any(a != t and a in anc(t) for a in h) for t in g):
            return True
    return False


def active_evidence(kg, evidence):
    """The evidence actually counted: hierarchy-implied findings are dropped."""
    pres = [g for g, e in evidence.items() if e["status"] == "present"]
    abse = [g for g, e in evidence.items() if e["status"] == "absent"]
    out = {}
    for g, e in evidence.items():
        others = pres if e["status"] == "present" else abse
        if not _implied(kg, g, others, e["status"]):
            out[g] = e
    return out


def log_likelihood(kg, evidence):
    logp = np.zeros(kg.n)
    for g, e in active_evidence(kg, evidence).items():
        if e["status"] == "present":
            logp += e["weight"] * np.log(kg.p_present(g))
        else:                         # no partial credit: denying the specific form says nothing
            logp += e["weight"] * ABSENT_WEIGHT * np.log(1 - kg.p_present(g, partial=False))
    return logp


# --------------------------------------------------------------------------- #
# Bayesian scoring
# --------------------------------------------------------------------------- #
def _entropy(p):
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum())


def _normalise(logp):
    p = np.exp(logp - logp.max())
    return p / p.sum()


def expected_gain(kg, post, groups):
    """Expected information gain (bits) of observing each group, under the current posterior."""
    if not groups:
        return np.array([])
    support = post > 1e-9
    pi = post[support] / post[support].sum()
    h0 = _entropy(pi)
    out = []
    for start in range(0, len(groups), EIG_BATCH):
        L = np.stack([kg.p_present(g)[support] for g in groups[start:start + EIG_BATCH]])
        p_yes = L @ pi
        for row, py in zip(L, p_yes):
            yes = row * pi / py
            no = (1 - row) * pi / (1 - py)
            out.append(h0 - (py * _entropy(yes) + (1 - py) * _entropy(no)))
    return np.array(out)


def _lse(x):
    """log(sum(exp(x))), stable."""
    m = float(np.max(x))
    return m + float(np.log(np.exp(x - m).sum()))


def log2_mass(logp, cols):
    """log2 of the posterior mass of `cols`, from the unnormalised log posterior."""
    return (_lse(logp[cols]) - _lse(logp)) / np.log(2)


def log_odds_vs_best_other(logp, cols):
    """log2 [ P(true diagnosis) / P(most probable other disease) ]."""
    others = np.ones(len(logp), dtype=bool)
    others[cols] = False
    if not others.any():
        return None
    return (_lse(logp[cols]) - float(logp[others].max())) / np.log(2)


def mid_rank(post, cols):
    """Best rank among the true columns; ties share their mid-rank (no index-order bias)."""
    best = None
    for c in cols:
        pt = post[c]
        tol = max(pt * 1e-9, 1e-300)
        above = int((post > pt + tol).sum())
        tied = int((np.abs(post - pt) <= tol).sum())
        r = above + (tied + 1) / 2
        best = r if best is None else min(best, r)
    return int(best) if float(best).is_integer() else best


def score_case(kg, steps, findings, links, true_cols):
    evidence = {}                                  # group -> {"status", "step", "weight"}
    detail = []

    def snapshot():
        logp = log_likelihood(kg, evidence)
        post = _normalise(logp)
        return (post, _entropy(post), mid_rank(post, true_cols), float(post[true_cols].sum()),
                logp)

    def observations(step_no):
        obs, notes = [], []
        for f in findings.get(step_no, []):
            g, how = resolve(kg, links.get(_fkey(f)), f["status"])
            notes.append({"term": f["term"], "status": f["status"], "how": how,
                          "hpo": sorted(g) if g else None,
                          "name": kg.group_name(g) if g else None,
                          "from_step": f.get("from_step", step_no)})
            if g is not None and (g, f["status"]) not in obs:
                obs.append((g, f["status"]))
        return obs, notes

    # step 0: what the doctor was given (referral, findings stated in the questions)
    obs0, notes0 = observations(0)
    for g, st in obs0:
        evidence[g] = {"status": st, "step": 0, "weight": 1.0}
    post, h_prior_given, rank, mass, logp = snapshot()
    h_uniform = float(np.log2(kg.n))
    if obs0:
        detail.append({"step": 0, "turn": 0, "kind": "given", "action": "referral / stated",
                       "answer": "", "findings": notes0, "mapped": True, "redundant": False,
                       "entropy_before": round(h_uniform, 4),
                       "info_gain_bits": round(h_uniform - h_prior_given, 4),
                       "eig_asked_bits": None, "eig_best_bits": None, "best_question": None,
                       "saturated": False, "true_rank": rank, "true_posterior": round(mass, 6),
                       "log_odds_true": _r(log_odds_vs_best_other(logp, true_cols), 3)})

    for s in steps:
        if s["kind"] == "given":
            continue
        obs, notes = observations(s["step"])
        new = [(g, st) for g, st in obs
               if g not in evidence or evidence[g]["status"] != st]
        redundant = bool(obs) and not new
        h_before = _entropy(post)

        new_groups = list(dict.fromkeys(g for g, _ in new))
        eig_new = expected_gain(kg, post, new_groups)
        eig_asked = float(eig_new.max()) if len(new_groups) else (0.0 if redundant else None)

        known_terms = {t for g in evidence for t in g}
        top = np.argsort(-post, kind="stable")[:TOP_DISEASES_FOR_BEST]
        cand = list(dict.fromkeys(
            frozenset([t]) for c in top for t in kg.disease_terms(int(c))
            if t not in known_terms and kg.member(t).mean() <= MAX_TERM_PREVALENCE))
        pool = cand + [g for g in new_groups if g not in cand]
        eig_pool = expected_gain(kg, post, pool)
        if len(pool):
            j = int(eig_pool.argmax())
            eig_best, best_q = float(eig_pool[j]), kg.group_name(pool[j])
        else:
            eig_best, best_q = None, None
        saturated = h_before < SATURATED_BITS

        weight = len(new) ** -DAMPING if new else 1.0
        for g, st in new:
            evidence[g] = {"status": st, "step": s["step"], "weight": weight}
        post, h_after, rank, mass, logp = snapshot()
        detail.append({
            "step": s["step"], "turn": s["turn"], "kind": s["kind"],
            "action": s["action"], "answer": s["answer"],
            "image_reading": bool(s.get("image_reading")),
            "findings": notes,
            "mapped": bool(new), "redundant": redundant,
            "entropy_before": round(h_before, 4),
            "info_gain_bits": round(h_before - h_after, 4),
            "eig_asked_bits": None if eig_asked is None else round(eig_asked, 4),
            "eig_best_bits": None if eig_best is None else round(eig_best, 4),
            "best_question": best_q, "saturated": saturated,
            "true_rank": rank, "true_posterior": round(mass, 6),
            "log_odds_true": _r(log_odds_vs_best_other(logp, true_cols), 3),
        })

    top5 = [{"id": kg.ids[c], "name": kg.names[c], "p": round(float(post[c]), 4)}
            for c in np.argsort(-post, kind="stable")[:5]]
    active = active_evidence(kg, evidence)
    used = [{"hpo": sorted(g), "name": kg.group_name(g), "status": e["status"], "step": e["step"],
             "counted": g in active} for g, e in evidence.items()]
    return {"detail": detail, "h_uniform": h_uniform, "h_start": h_prior_given,
            "h_end": _entropy(post), "top5": top5, "rank": rank, "mass": mass, "evidence": used,
            "post": post, "logp": logp, "groups": list(evidence)}


def _r(x, nd):
    return None if x is None else round(float(x), nd)


# --------------------------------------------------------------------------- #
# Per case and over the whole results file
# --------------------------------------------------------------------------- #
def _key(result):
    payload = json.dumps([PROMPT_VERSION, result.get("case"), result.get("transcript"),
                          referral_text(result), result.get("true_diagnosis"),
                          MODEL or sim.EXAMINER_MODEL])
    return hashlib.sha1(payload.encode()).hexdigest()


def _rank_band(rank, n):
    if rank is None:
        return None
    for k in (1, 5, 10, 50):
        if rank <= k:
            return f"top{k}"
    return "top1%" if rank <= 0.01 * n else "beyond1%"


def clean_doctor_diagnosis(text):
    """The disease name in a doctor's diagnosis line: no markdown, no justification in brackets,
    no lead-in such as "I propose a diagnosis of"."""
    t = re.sub(r"[*_`#]+", "", str(text or ""))
    for _ in range(3):                                  # nested brackets
        t = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", t)
    t = re.split(r"\s+(?:given|because|based on|due to the|as evidenced|supported by)\s+", t,
                 maxsplit=1, flags=re.I)[0]
    t = re.sub(r"^.*?\bdiagnos(?:is|e)\s*(?:of\b|is\b|:)\s*", "", t, flags=re.I)
    return " ".join(t.split()).strip(" .,:;-")


def doctor_diagnosis_text(result):
    """The doctor's final diagnosis: the parsed field, else the DIAGNOSIS part of the raw line."""
    dx = str(result.get("doctor_diagnosis") or "").strip()
    if not dx:
        raw = str(result.get("doctor_conclusion_raw") or "")
        m = re.search(r"DIAGNOSIS:\s*(.*?)(?:\|\||$)", raw, re.S)
        dx = m.group(1).strip() if m else ""
    return dx


def load_coverage(path, graph_key):
    """{case id: {"status", "match", "ids"}} for one graph, from kg_coverage.json; {} if absent."""
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for r in data.get("cases") or []:
        g = r.get(graph_key)
        if r.get("case") and isinstance(g, dict):
            out[str(r["case"])] = g
    print(f"[kg] coverage from {path}: {len(out)} cases, "
          f"{sum(g.get('status') == 'covered' for g in out.values())} covered in {graph_key}")
    return out


def _dx_from_coverage(kg, cov):
    """A diagnosis link in kg_eval's format from a kg_coverage row; None if unusable."""
    if not cov or cov.get("status") == "mapping_failed":
        return None
    ids = [x for x in cov.get("ids") or [] if x in kg.all_index]
    if len(ids) != len(cov.get("ids") or []):
        return None                                   # graph version differs: relink
    match = cov.get("match")
    return {"exact": ids if match == "exact" else [], "broader": ids if match == "broader" else [],
            "note": "from kg_coverage: " + str(cov.get("note") or "")}


def _first_step_within(detail, k):
    """Step number at which the true diagnosis first enters the top-k (0 = already given)."""
    for d in detail:
        if d["true_rank"] is not None and d["true_rank"] <= k:
            return d["step"]
    return None


def evaluate_case(kg, result, cache, coverage=None):
    key = _key(result)
    steps = parse_transcript(result.get("transcript") or [], referral_text(result))
    diagnosis = result.get("true_diagnosis") or ""
    doctor_dx = doctor_diagnosis_text(result)
    verdict = str(result.get("verdict") or "").strip().lower()
    asks = sum(s["kind"] == "ask" for s in steps)
    tests = sum(s["kind"] == "test" for s in steps)
    row = {
        "case": result.get("case"),
        "doctor_model": (result.get("models") or {}).get("doctor"),
        "true_diagnosis": diagnosis,
        "doctor_diagnosis": doctor_dx,
        "verdict": verdict,
        "graph": kg.key,
        "n_diseases": kg.n,
        "n_steps": asks + tests, "n_ask": asks, "n_test": tests,
        "has_referral": any(s["kind"] == "given" for s in steps),
        "n_image_readings": sum(bool(s.get("image_reading")) for s in steps),
    }
    c = cache.setdefault(key, {})
    dx_key, links_key = f"dx_{kg.key}", f"links_{kg.key}"
    doc_key = f"doctor_dx4_{kg.key}"                  # v4: cleaned text, own prompt, lenient ids
    if kg.key == "primekg":                           # entries written before graphs were split
        for old, new in (("dx", dx_key), ("links", links_key)):
            if old in c and new not in c:
                c[new] = c[old]

    def failed(what, exc):
        row.update({"kg_coverage": "mapping_failed", "error_type": "mapping_failed",
                    "kg_match": None, "kg_diagnosis_ids": [], "kg_diagnosis_names": [],
                    "kg_diagnosis_note": f"LLM call failed ({what}): {exc}"})
        if not c:
            cache.pop(key, None)
        return row

    # 1. true diagnosis -> graph: from kg_coverage when available, else LLM (cached)
    link = _dx_from_coverage(kg, (coverage or {}).get(str(result.get("case"))))
    if link is None:
        if dx_key not in c:
            try:
                c[dx_key] = link_diagnosis(kg, diagnosis, diagnosis_names(diagnosis))
            except MappingFailed as exc:
                return failed("diagnosis", exc)
        link = c[dx_key]

    match = "exact" if link["exact"] else "broader" if link["broader"] else None
    linked = link[match] if match else []
    n_ph = {x: kg.n_phenotypes(x) for x in linked}
    scorable = [x for x in linked if n_ph[x] >= MIN_DX_PHENOTYPES]
    if not match:
        coverage_status = "not_in_graph"
    elif not any(n_ph.values()):
        coverage_status = "no_phenotypes"     # the node exists, but the graph lists no phenotype
    elif not scorable:
        coverage_status = "few_phenotypes"    # too few relations to rank it meaningfully
    else:
        coverage_status = "covered"
    row.update({
        "kg_coverage": coverage_status, "kg_match": match,
        "kg_diagnosis_ids": linked,
        "kg_diagnosis_names": [kg.all_name(x) for x in linked],
        "kg_diagnosis_phenotypes": [n_ph[x] for x in linked],
        "kg_diagnosis_note": link.get("note", ""),
    })
    if coverage_status != "covered":                 # no further LLM call for uncovered cases
        row["error_type"] = "kg_not_covered"
        if not c:
            cache.pop(key, None)
        return row

    # 2. findings and term links (findings are shared across graphs, links are not)
    try:
        if "findings" not in c:
            c["findings"] = {str(k): v for k, v in extract_findings(steps).items()}
        findings = {int(k): v for k, v in c["findings"].items()}
        if links_key not in c:
            c[links_key] = link_terms(kg, findings)
    except MappingFailed as exc:
        return failed("findings", exc)
    links = c[links_key]

    # 3. doctor's diagnosis -> graph (only needed for regret and discriminating findings)
    doctor_dx_clean = clean_doctor_diagnosis(doctor_dx)
    if verdict == "correct":                          # judged correct: same entity as the truth
        doc_status = None
        doc_link = {"exact": list(scorable), "broader": [],
                    "note": "verdict correct: the true diagnosis' nodes are used"}
    elif not doctor_dx_clean:
        doc_status, doc_link = "none", {"exact": [], "broader": [], "note": ""}
    else:
        if doc_key not in c:
            try:
                c[doc_key] = link_diagnosis(
                    kg, doctor_dx_clean, diagnosis_names(doctor_dx_clean, split_commas=False),
                    label="DIAGNOSIS PROPOSED BY THE PHYSICIAN", prompt=DOCTOR_PICK_PROMPT,
                    extra_ids=linked)
            except MappingFailed:
                c.pop(doc_key, None)
        doc_link = c.get(doc_key)
        if doc_link is None:
            doc_status, doc_link = "mapping_failed", {"exact": [], "broader": [], "note": ""}
        else:
            doc_status = None
    doc_match = "exact" if doc_link["exact"] else "broader" if doc_link["broader"] else None
    doc_linked = doc_link[doc_match] if doc_match else []
    doc_scorable = [x for x in doc_linked if x in kg.col]
    if doc_status is None:
        doc_status = ("not_in_graph" if not doc_match else
                      "no_phenotypes" if not doc_scorable else "linked")

    # 4. Bayesian scoring
    cols = [kg.col[x] for x in scorable]
    sc = score_case(kg, steps, findings, links, cols)
    post, logp = sc["post"], sc["logp"]
    detail = [d for d in sc["detail"] if d["kind"] != "given"]
    ask_d = [d for d in detail if d["kind"] == "ask"]
    mapped = [d for d in detail if d["mapped"]]
    judged = [d for d in mapped if not d["saturated"]]
    ratios = [d["eig_asked_bits"] / d["eig_best_bits"] for d in judged
              if d["eig_best_bits"] and d["eig_asked_bits"] is not None]   # <= 1 by construction
    all_notes = [n for d in sc["detail"] for n in d["findings"]]
    mean = (lambda xs: round(float(np.mean(xs)), 4) if xs else None)
    frac = (lambda a, b: round(a / b, 3) if b else None)
    rank = sc["rank"]
    ranks = [d["true_rank"] for d in detail]

    if verdict == "correct":
        error_type = "correct"
    elif rank <= REACH_K:
        error_type = "synthesis_error"
    elif rank > UNREACH_K:
        error_type = "gathering_error"
    else:
        error_type = "undetermined"

    # A2 diagnostic regret, B2 discriminating findings (need the doctor's diagnosis)
    regret = doc_rank = doc_post = n_disc = None
    disc_names, same_as_true = [], bool(set(doc_scorable) & set(scorable))
    if doc_status == "linked":
        mcols = [kg.col[x] for x in doc_scorable]
        regret = log2_mass(logp, cols) - log2_mass(logp, mcols)
        doc_rank = mid_rank(post, mcols)
        doc_post = float(post[mcols].sum())
        if not same_as_true:
            v_star = cols[int(np.argmax(post[cols]))]
            m_star = mcols[int(np.argmax(post[mcols]))]
            disc = [g for g in sc["groups"]
                    if abs(kg.p_present(g)[v_star] - kg.p_present(g)[m_star]) >= DISC_TAU]
            n_disc, disc_names = len(disc), [kg.group_name(g) for g in disc]

    # B3 log-odds of the true diagnosis, C1 information regret, C2 convergence
    lo = [d["log_odds_true"] for d in detail if d["log_odds_true"] is not None]
    regrets = [d["eig_best_bits"] - d["eig_asked_bits"] for d in judged
               if d["eig_best_bits"] is not None and d["eig_asked_bits"] is not None]

    row.update({
        "n_findings": len(all_notes),
        "n_findings_linked": sum(n["how"] in ("linked", "backoff") for n in all_notes),
        "n_findings_backoff": sum(n["how"] == "backoff" for n in all_notes),
        "n_findings_given": sum(n.get("from_step", 0) != 0 for n in
                                (sc["detail"][0]["findings"] if sc["detail"]
                                 and sc["detail"][0]["kind"] == "given" else [])),
        "n_observed_findings": len(sc["groups"]),
        "mapped_fraction": frac(len(mapped), len(detail)),
        "unmapped_ask_fraction": frac(sum(not d["mapped"] for d in ask_d), len(ask_d)),
        "redundant_fraction": frac(sum(d["redundant"] for d in detail), len(detail)),
        "low_value_fraction": frac(sum(r < LOW_EIG_RATIO for r in ratios), len(ratios)),
        "saturated_steps": sum(d["saturated"] for d in mapped),
        "mean_info_gain_bits": mean([d["info_gain_bits"] for d in mapped]),
        "mean_eig_asked_bits": mean([d["eig_asked_bits"] for d in judged]),
        "mean_eig_ratio": mean(ratios),
        "info_regret_bits": round(float(sum(regrets)), 4) if regrets else None,
        "given_info_bits": round(sc["h_uniform"] - sc["h_start"], 3),
        "total_info_bits": round(sc["h_start"] - sc["h_end"], 3),
        "final_true_rank": rank,
        "min_true_rank": min(ranks) if ranks else rank,
        "final_rank_percentile": round(100 * rank / kg.n, 2),
        "final_rank_band": _rank_band(rank, kg.n),
        "final_true_posterior": round(sc["mass"], 6),
        "final_log_odds_true": lo[-1] if lo else None,
        "max_log_odds_true": max(lo) if lo else None,
        **{f"steps_to_top{k}": _first_step_within(sc["detail"], k) for k in CONVERGENCE_KS},
        "doctor_dx_clean": doctor_dx_clean,
        "doctor_dx_status": doc_status,
        "doctor_dx_note": doc_link.get("note", ""),
        "doctor_dx_match": doc_match,
        "doctor_dx_names": [kg.all_name(x) for x in doc_linked],
        "doctor_matches_true": same_as_true if doc_status == "linked" else None,
        "doctor_dx_rank": doc_rank,
        "doctor_dx_posterior": None if doc_post is None else round(doc_post, 6),
        "diagnostic_regret_bits": _r(regret, 3),
        "n_discriminating": n_disc,
        "discriminating_findings": disc_names,
        "reachable": rank <= REACH_K,
        "error_type": error_type,
        "kg_top5": sc["top5"],
        "rank_trajectory": ranks,
        "log_odds_trajectory": [d["log_odds_true"] for d in detail],
        "evidence": sc["evidence"],
        "steps": sc["detail"],
    })
    return row


CSV_FIELDS = ["case", "case_key", "doctor_model", "true_diagnosis", "doctor_diagnosis", "verdict", "graph",
              "kg_coverage", "kg_match", "kg_diagnosis_names", "kg_diagnosis_phenotypes",
              "n_diseases", "n_steps", "n_ask", "n_test", "has_referral", "n_image_readings",
              "n_findings", "n_findings_linked", "n_findings_backoff", "n_findings_given",
              "n_observed_findings",
              "mapped_fraction", "unmapped_ask_fraction", "redundant_fraction",
              "low_value_fraction", "saturated_steps", "mean_info_gain_bits",
              "mean_eig_asked_bits", "mean_eig_ratio", "info_regret_bits",
              "given_info_bits", "total_info_bits",
              "final_true_rank", "min_true_rank", "final_rank_percentile", "final_rank_band",
              "final_true_posterior", "final_log_odds_true", "max_log_odds_true",
              *[f"steps_to_top{k}" for k in CONVERGENCE_KS],
              "doctor_dx_clean", "doctor_dx_status", "doctor_dx_note", "doctor_dx_match", "doctor_dx_names", "doctor_matches_true",
              "doctor_dx_rank", "doctor_dx_posterior", "diagnostic_regret_bits",
              "n_discriminating", "reachable", "error_type"]

MEDIAN_FIELDS = ("mapped_fraction", "unmapped_ask_fraction", "redundant_fraction",
                 "low_value_fraction", "mean_info_gain_bits", "mean_eig_asked_bits",
                 "mean_eig_ratio", "info_regret_bits", "total_info_bits", "final_true_rank",
                 "min_true_rank", "final_rank_percentile", "final_log_odds_true",
                 "max_log_odds_true", *[f"steps_to_top{k}" for k in CONVERGENCE_KS],
                 "diagnostic_regret_bits", "n_discriminating")


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 2) if xs else None


def summarise(rows):
    def count(rs, key):
        out = {}
        for r in rs:
            out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
        return out

    def block(rs):
        wrong = [r for r in rs if r["verdict"] != "correct"]
        reg = [r.get("diagnostic_regret_bits") for r in wrong
               if r.get("diagnostic_regret_bits") is not None]
        return {
            "cases": len(rs),
            "error_types": count(rs, "error_type"),
            "verdicts": count(rs, "verdict"),
            "rank_bands": count(rs, "final_rank_band"),
            "doctor_dx_status": count(rs, "doctor_dx_status"),
            # calibration: if the graph is informative, correct cases rank much better
            "median_rank_percentile_correct": _median(
                [r.get("final_rank_percentile") for r in rs if r["verdict"] == "correct"]),
            "median_rank_percentile_incorrect": _median(
                [r.get("final_rank_percentile") for r in wrong]),
            # among wrong answers: did the data favour the true diagnosis (> 0) or the doctor's?
            "regret_sign_incorrect": {"favours_true": sum(x > 0 for x in reg),
                                      "favours_doctor": sum(x < 0 for x in reg),
                                      "tie": sum(x == 0 for x in reg)},
            "anchoring_incorrect": sum(r.get("n_discriminating") == 0 for r in wrong),
            "never_reached": {f"top{k}": sum(r.get(f"steps_to_top{k}") is None for r in rs)
                              for k in CONVERGENCE_KS},
            "median": {k: _median([r.get(k) for r in rs]) for k in MEDIAN_FIELDS},
        }

    covered = [r for r in rows if r["kg_coverage"] == "covered"]
    by_match = {}
    for r in covered:
        by_match.setdefault(r["kg_match"], []).append(r)
    return {"cases": len(rows), "covered": len(covered),
            "coverage": round(len(covered) / len(rows), 3) if rows else None,
            "coverage_status": count(rows, "kg_coverage"),
            "error_types": count(rows, "error_type"),
            "all_covered": block(covered),
            "by_match": {k: block(v) for k, v in sorted(by_match.items())}}


def settings(kg=None):
    return {"model": MODEL or sim.EXAMINER_MODEL, "graph": kg.label if kg else GRAPH,
            "prompt_version": PROMPT_VERSION, "p_hit": P_HIT, "p_background": P_BACKGROUND,
            "p_excluded": P_EXCLUDED, "p_partial": P_PARTIAL,
            "partial_max_prevalence": PARTIAL_MAX_PREVALENCE, "absent_weight": ABSENT_WEIGHT,
            "damping": DAMPING, "min_dx_phenotypes": MIN_DX_PHENOTYPES,
            "reach_k": REACH_K, "unreach_k": UNREACH_K, "low_eig_ratio": LOW_EIG_RATIO,
            "saturated_bits": SATURATED_BITS, "max_term_prevalence": MAX_TERM_PREVALENCE,
            "backoff_max_prevalence": BACKOFF_MAX_PREVALENCE, "disc_tau": DISC_TAU,
            "convergence_ks": list(CONVERGENCE_KS), "referral_keys": list(REFERRAL_KEYS)}


def _atomic_write(path, text):
    """Write via a temporary file and rename: an interruption cannot leave a truncated file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _csv_row(r):
    return {**r,
            "kg_diagnosis_names": "; ".join(r.get("kg_diagnosis_names") or []),
            "kg_diagnosis_phenotypes": "; ".join(
                str(x) for x in r.get("kg_diagnosis_phenotypes") or []),
            "doctor_dx_names": "; ".join(r.get("doctor_dx_names") or [])}


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if r["kg_coverage"] == "covered":
                w.writerow(_csv_row(r))


def evaluate(results_path, model=None, kg_path=KG_PATH, out_json="kg_eval.json",
             out_csv="kg_eval.csv", cache_path="kg_cache.json", kg=None, graph=None,
             coverage_path=COVERAGE_PATH, resume=True):
    """Score every case in results.json. Returns the summary; writes the JSON, CSV and cache.

    Only cases covered by the graph are written to the CSV and to the JSON "cases"; the summary
    still reports how many cases were not covered and why.

    Resumable: every finished case is appended to <out_json>l (JSON lines) and to the CSV. On a
    new run with resume=True the cases already there are skipped, so an interrupted run goes on
    where it stopped. resume=False starts from scratch (the LLM cache is kept either way)."""
    global MODEL
    MODEL = model or MODEL
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    if isinstance(results, dict):
        results = [results]
    cache_file = Path(cache_path)
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    kg = kg or load_graph(HPOVocab(), graph, kg_path)
    coverage = load_coverage(coverage_path, kg.key)

    # progress file: first line = settings, then one JSON row per finished case
    progress = Path(str(out_json) + "l")
    run_settings = settings(kg)
    rows = []
    if resume and progress.exists():
        lines = [json.loads(x) for x in progress.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
        if lines and lines[0].get("_settings") not in (None, run_settings):
            raise ValueError(
                f"{progress} was produced with different settings; to start again use "
                "resume=False (or delete it), or restore the previous settings.")
        rows = [x for x in lines[1:] if "case_key" in x]
        print(f"[kg] resuming: {len(rows)} cases already done in {progress}")
    else:
        progress.write_text(json.dumps({"_settings": run_settings}) + "\n", encoding="utf-8")
    done = {r["case_key"] for r in rows}
    _write_csv(out_csv, rows)                        # CSV consistent with the progress file

    fh = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    try:
        with open(progress, "a", encoding="utf-8") as pf:
            for n, result in enumerate(results, 1):
                key = _key(result)
                if key in done:
                    continue
                row = evaluate_case(kg, result, cache, coverage)
                row["case_key"] = key
                _atomic_write(cache_file, json.dumps(cache, indent=1, ensure_ascii=False))
                _print_row(n, len(results), row, kg)
                if row["kg_coverage"] == "mapping_failed":
                    continue                          # not recorded: retried on the next run
                rows.append(row)
                done.add(key)
                pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                pf.flush()
                if row["kg_coverage"] == "covered":   # only cases covered by the graph
                    writer.writerow(_csv_row(row))
                    fh.flush()
    finally:
        fh.close()

    summary = summarise(rows)
    covered = [r for r in rows if r["kg_coverage"] == "covered"]
    _atomic_write(out_json, json.dumps({"settings": run_settings, "summary": summary,
                                        "cases": covered}, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(summary, indent=2))
    print(f"{len(covered)} covered cases saved (of {len(rows)} in the results file)")
    print(f"Saved {out_json}, {out_csv} (progress in {progress})")
    return summary


def _print_row(n, total, row, kg):
    name = Path(str(row["case"])).name
    if row["kg_coverage"] == "covered":
        print(f"[{n}/{total}] {name}: {row['error_type']} | {row['kg_match']} "
              f"({'; '.join(row['kg_diagnosis_names'])}) | rank {row['final_true_rank']}"
              f"/{row['n_diseases']} (min {row['min_true_rank']}) | EIG ratio "
              f"{row['mean_eig_ratio']} | regret {row['diagnostic_regret_bits']} "
              f"(doctor dx: {row['doctor_dx_status']})")
    elif row["kg_coverage"] == "mapping_failed":
        print(f"[{n}/{total}] {name}: mapping failed ({row['kg_diagnosis_note']}); "
              "will retry next run")
    elif VERBOSE_UNCOVERED:
        what = {"no_phenotypes": f"in {kg.label} without phenotypes",
                "few_phenotypes": f"in {kg.label} with < {MIN_DX_PHENOTYPES} phenotypes",
                "not_in_graph": f"not in {kg.label}"}[row["kg_coverage"]]
        names = "; ".join(row["kg_diagnosis_names"])
        print(f"[{n}/{total}] {name}: {what}"
              + (f": {names}" if names else "") + f" ({row['true_diagnosis']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results", help="results.json produced by the arena")
    p.add_argument("--model", default=None, help="mapping model, e.g. genai:gemini-3.5-flash-lite")
    p.add_argument("--graph", default=GRAPH, choices=["hpo", "primekg"])
    p.add_argument("--coverage", default=COVERAGE_PATH,
                   help="kg_coverage.json from kg_coverage.py ('' to relink every diagnosis)")
    p.add_argument("--kg", default=KG_PATH, help="PrimeKG kg.csv (only with --graph primekg)")
    p.add_argument("--out", default="kg_eval.json")
    p.add_argument("--csv", default="kg_eval.csv")
    p.add_argument("--cache", default="kg_cache.json")
    p.add_argument("--reach-k", type=int, default=REACH_K)
    p.add_argument("--unreach-k", type=int, default=UNREACH_K)
    p.add_argument("--p-hit", type=float, default=P_HIT)
    p.add_argument("--damping", type=float, default=DAMPING)
    p.add_argument("--p-partial", type=float, default=P_PARTIAL)
    p.add_argument("--absent-weight", type=float, default=ABSENT_WEIGHT)
    p.add_argument("--min-dx-phenotypes", type=int, default=MIN_DX_PHENOTYPES)
    p.add_argument("--restart", action="store_true",
                   help="ignore the progress file and start from scratch (cache kept)")
    a = p.parse_args()
    REACH_K, UNREACH_K = a.reach_k, max(a.unreach_k, a.reach_k)
    P_HIT, DAMPING, MIN_DX_PHENOTYPES = a.p_hit, a.damping, a.min_dx_phenotypes
    P_PARTIAL, ABSENT_WEIGHT = a.p_partial, a.absent_weight
    evaluate(a.results, a.model, a.kg, a.out, a.csv, a.cache, graph=a.graph,
             coverage_path=a.coverage, resume=not a.restart)