import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvidenceAnnotator:
    def __init__(self, questions_path, chunks_dir):
        self.questions_path = questions_path
        self.chunks_dir = chunks_dir
        self.dataset = read_json(questions_path)
        self.manifest = read_json(chunks_dir / "manifest.json")
        self.by_source = defaultdict(list)
        seen = set()
        with (chunks_dir / "chunks.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                chunk = json.loads(line)
                if chunk["id"] in seen or not chunk["text"].strip():
                    raise ValueError(f"Duplicate or empty chunk: {chunk['id']}")
                seen.add(chunk["id"])
                self.by_source[chunk["source_path"]].append(chunk)

    def match_evidence(self, evidence):
        quote = evidence["quote"].split()
        if not quote:
            raise ValueError(f"Empty evidence: {evidence['id']}")
        pattern = re.compile(r"\s+".join(map(re.escape, quote)))
        matches = []
        for chunk in self.by_source[evidence["source_path"]]:
            match = pattern.search(chunk["text"])
            if match is None:
                continue
            if chunk["source_commit"] != self.dataset["corpus"]["source_commit"]:
                raise ValueError(f"Source commit mismatch: {chunk['id']}")
            matches.append({
                "chunk_id": chunk["id"],
                "section_id": chunk["section_id"],
                "heading_path": chunk["heading_path"],
                "span_start": match.start(),
                "span_end": match.end(),
            })
        if not matches:
            raise ValueError(f"Evidence not found in chunks: {evidence['id']}")
        return {"evidence_id": evidence["id"], "matches": matches}

    def annotate_question(self, question):
        sources = {e["source_path"] for e in question["evidence"]}
        if len(sources) != question["required_source_count"]:
            raise ValueError(f"Source count mismatch: {question['id']}")
        if question["type"] not in {"single_hop", "multi_hop"}:
            raise ValueError(f"Unknown question type: {question['id']}")
        groups = [self.match_evidence(e) for e in question["evidence"]]
        chunk_sets = [{m["chunk_id"] for m in g["matches"]} for g in groups]
        if not chunk_sets:
            raise ValueError(f"Question has no evidence: {question['id']}")
        if question["type"] == "multi_hop":
            if len(sources) < 2 or set.intersection(*chunk_sets):
                raise ValueError(f"Multi-hop needs separate sources: {question['id']}")
        return {
            "question_id": question["id"],
            "relevant_chunk_ids": sorted(set.union(*chunk_sets)),
            "required_evidence_ids": [g["evidence_id"] for g in groups],
            "evidence_groups": groups,
        }

    def annotate(self):
        questions = self.dataset["questions"]
        ids = [q["id"] for q in questions]
        evidence_ids = [e["id"] for q in questions for e in q["evidence"]]
        if len(ids) != len(set(ids)) or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Duplicate question or evidence IDs")
        judgments = [self.annotate_question(q) for q in questions]
        document_paths = {e["document_path"] for q in questions for e in q["evidence"]}
        return {
            "schema_version": 1,
            "chunk_version": self.chunks_dir.name,
            "questions_sha256": checksum(self.questions_path),
            "chunks_sha256": checksum(self.chunks_dir / "chunks.jsonl"),
            "manifest_sha256": checksum(self.chunks_dir / "manifest.json"),
            "corpus": self.dataset["corpus"],
            "chunking": {k: self.manifest[k] for k in ("splitter", "config", "tokenizer")},
            "source_inputs": [i for i in self.manifest["inputs"] if i["path"] in document_paths],
            "relevance": {"1": "Contains a manually selected supporting excerpt"},
            "unlisted_chunks": "unjudged",
            "coverage_rule": "At least one matched chunk per required evidence ID",
            "judgments": judgments,
        }


def save_labels(labels, output):
    output.mkdir(parents=True, exist_ok=True)
    version = labels["chunk_version"]
    (output / f"{version}.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        f"{q['question_id']}\t0\t{chunk_id}\t1\n"
        for q in labels["judgments"] for chunk_id in q["relevant_chunk_ids"]
    ]
    (output / f"{version}.tsv").write_text("".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Bind annotated source excerpts to chunk IDs")
    parser.add_argument("--questions", type=Path, default=PROJECT / "evaluation/dataset/questions.json")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "evaluation/dataset/labels")
    args = parser.parse_args()
    labels = EvidenceAnnotator(args.questions, args.chunks.resolve()).annotate()
    save_labels(labels, args.output)
    pairs = sum(len(q["relevant_chunk_ids"]) for q in labels["judgments"])
    print(f"Annotated {len(labels['judgments'])} questions, {pairs} positive question/chunk pairs")


if __name__ == "__main__":
    main()
