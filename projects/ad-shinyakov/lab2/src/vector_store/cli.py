import argparse
from contextlib import closing
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .config import IndexConfig
from .loader import VectorLoader


def main(argv=None):
    cli = argparse.ArgumentParser(description="Create a Qdrant collection and load versioned embeddings")
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    cli.add_argument("--collection")
    cli.add_argument("--m", type=int, default=16)
    cli.add_argument("--ef-construction", type=int, default=128)
    cli.add_argument("--ef-search", type=int, default=128, help="Saved query default; does not rebuild HNSW")
    cli.add_argument("--batch-size", type=int, default=128)
    cli.add_argument("--timeout", type=int, default=60)
    cli.add_argument("--drop-and-reindex", "--rebuild", dest="reset", action="store_true")
    args = cli.parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        config = IndexConfig(args.m, args.ef_construction)
        with closing(QdrantClient(url=args.url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=args.timeout)) as client:
            manifest = VectorLoader(client, config).run(args.input, args.collection, args.batch_size, args.reset, args.ef_search)
        print(f"Ready: {manifest['collection']} ({manifest['count']} points)")
        return 0
    except (OSError, ValueError, RuntimeError, ResponseHandlingException, UnexpectedResponse) as error:
        print(str(error), file=sys.stderr)
        return 1
