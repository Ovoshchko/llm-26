import argparse
import sys
from pathlib import Path

from .pipeline import ChunkingPipeline
from .splitter import ChunkConfig, MarkdownChunker
from .tokenizer import TokenCounter


def parse_arguments(argv=None):
    cli = argparse.ArgumentParser(description="Split normalized Markdown into token-limited chunks by H1–H3.")
    cli.add_argument("--input", type=Path, default=Path("data/kubernetes"))
    cli.add_argument("--output", type=Path, default=Path("data/chunks"))
    cli.add_argument("--splitter", choices=["markdown", "recursive"], default="markdown", help="Split by Markdown sections or recursively over the whole document")
    cli.add_argument("--chunk-size", type=int, default=800)
    cli.add_argument("--overlap", type=int, default=100)
    cli.add_argument("--tokenizer", default="Qwen/Qwen3-Embedding-0.6B")
    headings = cli.add_mutually_exclusive_group()
    headings.add_argument("--headings", dest="include_headings", action="store_true", help="Add heading path to chunk text")
    headings.add_argument("--no-headings", dest="include_headings", action="store_false", help="Keep heading path only in metadata (default)")
    cli.set_defaults(include_headings=False)
    return cli.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    try:
        config = ChunkConfig(args.chunk_size, args.overlap, args.include_headings, args.splitter)
        tokenizer = TokenCounter(args.tokenizer)
        chunker = MarkdownChunker(tokenizer, config)
        destination, count = ChunkingPipeline(chunker, args.output).run(args.input)
        print(f"Saved {count} chunks to {destination}")
        return 0
    except (OSError, ValueError, NotImplementedError) as error:
        print(str(error), file=sys.stderr)
        return 1
