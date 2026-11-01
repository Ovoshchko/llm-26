import argparse
import sys
from pathlib import Path

from .config import EmbeddingConfig
from .model import SentenceEncoder
from .pipeline import EmbeddingPipeline


def main(argv=None):
    cli = argparse.ArgumentParser(description="Build reusable dense embeddings from chunks.jsonl")
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--output", type=Path, default=Path("data/embeddings"))
    cli.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    cli.add_argument("--revision", default="main")
    cli.add_argument("--device", choices=["cpu", "mps", "cuda"])
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument("--max-length", type=int, default=2048)
    cli.add_argument("--dimensions", type=int)
    cli.add_argument("--document-prefix", default="")
    cli.add_argument("--query-prefix", default=EmbeddingConfig.query_prefix)
    cli.add_argument("--rebuild", action="store_true")
    args = cli.parse_args(argv)
    try:
        config = EmbeddingConfig(args.model, args.revision, args.max_length, args.dimensions, args.document_prefix, args.query_prefix)
        encoder = SentenceEncoder(config, args.device)
        destination = EmbeddingPipeline(encoder, args.output).run(args.input, args.batch_size, args.rebuild)
        print(f"Embeddings: {destination}")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
