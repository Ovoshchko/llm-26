import re


class EvidenceLedger:
    def __init__(self, requirements, verify=True):
        self.requirements = requirements
        self.verify = verify
        self.entries = []
        self.reviews = {}
        self.approved = set()

    @property
    def missing(self):
        covered = {i for i, review in self.reviews.items() if review["status"] == "supported"}
        if not self.verify:
            covered = {e["requirement_id"] for e in self.entries}
        return [index for index in range(1, len(self.requirements) + 1) if index not in covered]

    def add(self, quotes, contexts):
        sources = {c["source_id"]: c for c in contexts}
        accepted, rejected = [], []
        for quote in quotes:
            source = sources.get(quote.source_id)
            reason = None
            if not 1 <= quote.requirement_id <= len(self.requirements):
                reason = "Unknown requirement"
            elif source is None:
                reason = "Unknown source"
            else:
                pattern = r"\s+".join(re.escape(word) for word in quote.quote.split())
                match = re.search(pattern, source["text"]) if pattern else None
                if match is None:
                    reason = "Quote is absent from source"
            if reason:
                rejected.append({**quote.model_dump(), "reason": reason})
                continue
            start = source["text"].rfind("\n\n", 0, match.start())
            start = 0 if start < 0 else start + 2
            end = source["text"].find("\n\n", match.end())
            end = len(source["text"]) if end == -1 else end
            entry = next((e for e in self.entries if e["requirement_id"] == quote.requirement_id
                          and e["chunk_id"] == source["chunk_id"]), None)
            if entry is not None:
                if entry["span_start"] <= start and end <= entry["span_end"]:
                    continue
                start, end = min(start, entry["span_start"]), max(end, entry["span_end"])
                self.approved.difference_update(e["evidence_id"] for e in self.entries
                                                if e["requirement_id"] == quote.requirement_id)
                self.reviews.pop(quote.requirement_id, None)
            else:
                entry = {**source, "evidence_id": len(self.entries) + 1,
                         "requirement_id": quote.requirement_id}
                self.entries.append(entry)
            entry.update(quote=source["text"][start:end], span_start=start, span_end=end)
            if not self.verify:
                self.approved.add(entry["evidence_id"])
            if entry not in accepted:
                accepted.append(entry)
        return accepted, rejected

    def review(self, requirement_id, review, sources):
        by_id = {s["source_id"]: s for s in sources}
        if review.requirement_id != requirement_id:
            raise ValueError("Review must address the requested requirement")
        if not set(review.source_ids).issubset(by_id):
            raise ValueError("Review references an unknown evidence source")
        if review.status == "supported" and not (review.answer and review.answer.strip()):
            raise ValueError("Supported verdict needs a grounded answer")
        if review.status == "supported" and not review.source_ids:
            raise ValueError("Supported verdict needs evidence sources")
        self.reviews[requirement_id] = {**review.model_dump(),
            "evidence_ids": [by_id[i]["evidence_id"] for i in review.source_ids]}
        if review.status == "supported":
            self.approved.update(by_id[i]["evidence_id"] for i in review.source_ids)

    def sources(self):
        sources = {}
        for entry in self.entries:
            if entry["evidence_id"] not in self.approved:
                continue
            key = (entry["chunk_id"], entry["span_start"], entry["span_end"])
            if key not in sources:
                sources[key] = {**entry, "text": entry["quote"], "requirement_ids": []}
            sources[key]["requirement_ids"].append(entry["requirement_id"])
        return list(sources.values())
