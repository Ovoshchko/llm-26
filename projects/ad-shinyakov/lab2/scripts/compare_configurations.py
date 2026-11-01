import argparse
from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from qdrant_client import QdrantClient
from evaluation.configurations import ConfigurationBenchmark


def main():
    parser = argparse.ArgumentParser(description="Compare stage 3–4 configurations: 1024/512/256-dimensional embeddings")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=PROJECT / "evaluation/dataset/questions.json")
    parser.add_argument("--labels", type=Path, default=PROJECT / "evaluation/dataset/labels/3cccc393ec5ee49d.json")
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--device")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or PROJECT / "evaluation/local_runs/embedding_dimensions" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with closing(QdrantClient(url=args.url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=60)) as client:
        benchmark = ConfigurationBenchmark(client, args.embeddings, args.chunks, args.questions, args.labels)
        result = benchmark.run(output, args.repetitions, args.device)
    print(f"Recommended: {result['recommended']}; report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
