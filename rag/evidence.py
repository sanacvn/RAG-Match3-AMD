"""Validate quoted grounding and explicit cross-file links; map IDs to paths in code."""
from collections import defaultdict
import heapq
import re
from .retrieval import tokens
from .schema import EMPTY, validate_output


def compact(text):
    return re.sub(r"\s+", "", text).casefold()


def complete_value(answer, quote):
    normalized = re.sub(r"\s+", " ", answer).casefold()
    pattern = re.escape(normalized).replace(r"\ ", r"\s+")
    if not re.search(r"(?<!\w)" + pattern + r"(?!\w)", quote.casefold()):
        return False
    # Reject truncated identifying qualifiers even when they are substrings.
    for value in re.findall(r"\bREV[- _][A-Z0-9.-]+|\bQ[1-4]\s+FY\s*\d{2,4}\b", quote, re.I):
        if compact(answer) in compact(value) and compact(answer) != compact(value):
            return False
    return True


def select_output(internal, candidates, query):
    def reject(reason):
        return dict(EMPTY), reason
    if not isinstance(internal, dict) or not isinstance(internal.get("answer"), str):
        return reject("schema")
    answer = internal["answer"].strip()
    if not answer:
        return reject("model_abstention")
    if "\n" in answer or len(answer) > 200:
        return reject("value_format")
    ids = internal.get("evidence_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        return reject("invalid_ids")
    lookup = {e.chunk_id: e for e in candidates}
    if any(i not in lookup for i in ids):
        return reject("unknown_evidence")
    proofs = internal.get("proofs")
    if not isinstance(proofs, list) or not all(isinstance(p, dict) for p in proofs):
        return reject("missing_proofs")
    quotes, roles = defaultdict(list), defaultdict(set)
    for p in proofs:
        i, quote, role = p.get("evidence_id"), p.get("quote"), p.get("role")
        if i not in ids or not isinstance(quote, str) or not quote.strip() or quote not in lookup[i].content:
            return reject("ungrounded_quote")
        if role not in {"anchor", "bridge", "value", "revision"}:
            return reject("invalid_role")
        quotes[i].append(quote)
        roles[i].add(role)
    if set(quotes) != set(ids):
        return reject("uncited_proof")
    values = {i for i in ids if roles[i] & {"anchor", "value"} and any(complete_value(answer, q) for q in quotes[i])}
    if not values:
        return reject("unsupported_value")
    stop = {"what", "which", "is", "the", "a", "an", "for", "of", "in", "on", "to", "does", "when", "value"}
    qterms = set(tokens(query))-stop
    anchors = {i for i in ids if "anchor" in roles[i] and qterms & set(tokens(" ".join(quotes[i])))}
    if not anchors:
        return reject("missing_question_anchor")
    links = internal.get("links", [])
    if not isinstance(links, list):
        return reject("invalid_links")
    graph = defaultdict(set)
    for link in links:
        if not isinstance(link, dict):
            return reject("invalid_link")
        a, b, identifier = link.get("from_id"), link.get("to_id"), link.get("identifier")
        if a not in ids or b not in ids or a == b or not isinstance(identifier, str) or not identifier.strip():
            return reject("invalid_link")
        # Exact identifiers across the quoted proof, not a match on a generic word.
        identifier_pattern = r"(?<![\w.-])" + re.escape(identifier.casefold()).replace(r"\ ", r"\s+") + r"(?![\w.-])"
        if not re.search(r"\d|[_#-]", identifier) or any(not re.search(identifier_pattern, " ".join(quotes[x]).casefold()) for x in (a, b)):
            return reject("unsupported_link")
        graph[a].add(b)
        graph[b].add(a)
    files = {lookup[i].source_path for i in ids}
    if len(files) > 1:
        # Every selected unit must be on a simple anchor-to-value path. Topic-only
        # leaves and duplicate value sources are rejected instead of over-cited.
        paths = []
        # Bounded state search avoids enumerating every simple path in a dense
        # graph. Prioritize fewer distinct files, then fewer evidence units.
        for a in anchors:
            initial_files = frozenset([lookup[a].source_path])
            heap = [(1, 1, (a,), a, initial_files)]
            visited = set()
            while heap:
                _, _, path, node, used_files = heapq.heappop(heap)
                state = (node, used_files)
                if state in visited:
                    continue
                visited.add(state)
                if node in values:
                    paths.append(list(path))
                    break
                for neighbor in graph[node]:
                    if neighbor not in path:
                        files_next = used_files | {lookup[neighbor].source_path}
                        heapq.heappush(heap, (len(files_next), len(path)+1, path+(neighbor,), neighbor, files_next))
        if not paths:
            return reject("incomplete_chain")
        # One minimal path is the proof of a single returned value. Every file on
        # that path remains necessary to this chain; reject optional extra files.
        chosen = min(paths, key=lambda p: (len({lookup[i].source_path for i in p}), len(p), p))
        required_files = {lookup[i].source_path for i in chosen}
        if required_files != files:
            return reject("unnecessary_source")
    for path in files:
        if path.startswith("/") or ".." in path.split("/"):
            return reject("unsafe_source_path")
    return validate_output({"answer": answer, "citations": sorted(files)}), "answered"
