from collections import Counter, defaultdict
import math
import re
import time

TOKEN = re.compile(r"[\w#]+(?:[-.][\w#]+)*", re.UNICODE)
IDENTIFIER = re.compile(r"\b(?:[A-Z][A-Z0-9_#]*[- ]\d+[A-Z0-9.-]*|[A-Z]+\d+[A-Z0-9.-]*|[A-Z][A-Z0-9_]*_[A-Z0-9_]+#?)\b")


def tokens(text):
    return [s.lower() for s in TOKEN.findall(text)]


def rrf(rankings, constant=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] += 1 / (constant + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


class Retriever:
    def __init__(self, evidence, config, models, vectors=None):
        self.evidence, self.config, self.models = evidence, config, models
        self.lengths = []
        self.postings = defaultdict(list)
        for i, e in enumerate(evidence):
            counts = Counter(tokens(e.content))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((i, count))
        self.avg_length = sum(self.lengths) / max(1, len(evidence))
        self.vectors = vectors

    def build_dense(self):
        if self.config.method != "bm25" and self.evidence:
            self.vectors = self.models.embed([e.content for e in self.evidence], query=False)

    def lexical(self, query, k):
        scores = defaultdict(float)
        total = len(self.evidence)
        for term in set(tokens(query)):
            posting = self.postings.get(term, [])
            df = len(posting)
            idf = math.log(1 + (total-df+.5)/(df+.5))
            for i, tf in posting:
                denom = tf + 1.2*(.25 + .75*self.lengths[i]/max(1, self.avg_length))
                scores[i] += idf * tf * 2.2 / denom
        return sorted(scores, key=lambda i: (-scores[i], i))[:k]

    def rank(self, query, deadline):
        c = self.config
        lexical = self.lexical(query, c.candidates) if c.method != "dense" else []
        dense = []
        if c.method != "bm25":
            import numpy as np
            vector = self.models.embed([query], query=True)[0]
            dense = np.argsort(-(self.vectors @ vector), kind="stable")[:c.candidates].tolist()
        return rrf([lexical, dense], c.rrf_c) if c.method == "hybrid" else lexical or dense

    def search(self, query, deadline):
        c = self.config
        if not self.evidence:
            return [], {"initial_ids": [], "expansion_identifiers": []}
        def base(q):
            return self.rank(q, deadline)
        ranked = base(query)
        initial = ranked[:c.context_k]
        identifiers = []
        if c.expand and time.monotonic() < deadline-3:
            seen = set(IDENTIFIER.findall(query))
            for i in initial:
                for identifier in IDENTIFIER.findall(self.evidence[i].content):
                    if identifier not in seen:
                        seen.add(identifier)
                        identifiers.append(identifier)
            identifiers = identifiers[:c.expansion_ids]
            # Exactly one bounded round. Keep anchor evidence to avoid losing bridges.
            follow = base(query + " " + " ".join(identifiers)) if identifiers else []
            # Pure identifier searches find rows that repeat few words from the question.
            follows = [base(identifier) for identifier in identifiers]
            ranked = rrf([ranked, follow] + follows, c.rrf_c)[:c.candidates]
        if c.rerank and time.monotonic() < deadline-2:
            scores = self.models.rerank(query, [self.evidence[i].content for i in ranked[:c.candidates]])
            ranked = [i for _, i in sorted(zip(scores, ranked), key=lambda p: -p[0])]
        selected = ranked[:c.context_k]
        if c.expand and identifiers:
            # Reserve up to two original anchor chunks; bound context size overall.
            anchors = initial[:min(2, c.context_k)]
            selected = anchors + [i for i in ranked if i not in anchors][:c.context_k-len(anchors)]
        return [self.evidence[i] for i in selected], {
            "initial_ids": [self.evidence[i].chunk_id for i in initial],
            "expansion_identifiers": identifiers,
            "candidate_ids": [self.evidence[i].chunk_id for i in ranked],
        }
