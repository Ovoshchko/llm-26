import argparse
import sys
from pathlib import Path

from .document import DocumentParser
from .download import RepositoryDownloader
from .images import ImageStore
from .pipeline import ParsingPipeline
from .source import RepositorySource
from .storage import FileStorage


def parse_arguments(argv=None):
    cli = argparse.ArgumentParser(description="Download and normalize Kubernetes Markdown documentation.")
    cli.add_argument("--repo", type=Path, help="Use a local checkout instead of downloading")
    cli.add_argument("--ref", default="release-1.35", help="Git branch or tag to download (default: release-1.35)")
    cli.add_argument("--cache", type=Path, default=Path("data/sources"), help="Downloaded repository cache")
    cli.add_argument("--section", action="append", help="Path relative to content/<language>; repeatable (default: docs/concepts)")
    cli.add_argument("--language", default="en")
    cli.add_argument("--base-url", help="Published documentation origin; inferred from --ref for downloads")
    cli.add_argument("--output", type=Path, default=Path("data/kubernetes"))
    cli.add_argument("--limit", type=int, help="Maximum number of Markdown files")
    args = cli.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        cli.error("--limit must be positive")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    try:
        repository = args.repo or RepositoryDownloader(args.cache).download(args.ref)
        base_url = args.base_url or "https://kubernetes.io"
        if not args.repo and not args.base_url and args.ref.startswith("release-"):
            base_url = f"https://v{args.ref.removeprefix('release-').replace('.', '-')}.docs.kubernetes.io"
        source = RepositorySource(repository, args.language, base_url)
        storage = FileStorage(args.output.resolve())
        if storage.root.is_relative_to(source.content):
            raise ValueError("Output must be outside the source content directory")
        images = ImageStore(source, storage)
        documents = DocumentParser(source, storage, images)
        return int(ParsingPipeline(source, documents, storage).run(args.section or ["docs/concepts"], args.limit))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
