import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_chunks(path):
    chunks, ids = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            if not isinstance(chunk, dict) or not isinstance(chunk.get("id"), str) or not chunk["id"]:
                raise ValueError(f"Invalid chunk id at line {number}")
            if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                raise ValueError(f"Empty chunk text at line {number}")
            if chunk["id"] in ids:
                raise ValueError(f"Duplicate chunk id: {chunk['id']}")
            ids.add(chunk["id"])
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No chunks in {path}")
    return chunks
