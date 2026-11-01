import argparse
from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from qdrant_client import QdrantClient
from evaluation.runner import RetrievalEvaluation


def main():
    parser = argparse.ArgumentParser(description="Compare dense, BM25 and hybrid RRF retrieval")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=PROJECT / "evaluation/dataset/questions.json")
    parser.add_argument("--labels", type=Path, default=PROJECT / "evaluation/dataset/labels/3cccc393ec5ee49d.json")
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection")
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or PROJECT / "evaluation/local_runs/search_methods" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with closing(QdrantClient(url=args.url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=60)) as client:
        runner = RetrievalEvaluation(client, args.embeddings, args.questions, args.labels, args.collection)
        report = runner.run(output, device=args.device)
    for result in report["variants"]:
        print(result["name"], result["metrics"]["all"])
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
