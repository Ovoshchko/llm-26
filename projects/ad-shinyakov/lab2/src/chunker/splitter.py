import hashlib
from dataclasses import dataclass
from itertools import groupby
from langchain_core.documents import Document
from langchain_text_splitters import ExperimentalMarkdownSyntaxTextSplitter, RecursiveCharacterTextSplitter


@dataclass
class ChunkConfig:
    chunk_size: int = 800
    overlap: int = 100
    include_headings: bool = False
    splitter: str = "markdown"

    def __post_init__(self):
        if self.splitter not in {"markdown", "recursive"}:
            raise ValueError("splitter must be markdown or recursive")
        if not 100 <= self.chunk_size <= 1000:
            raise ValueError("chunk_size must be between 100 and 1000")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must be nonnegative and smaller than chunk_size")


class MarkdownChunker:
    def __init__(self, tokenizer, config=None):
        self.tokenizer = tokenizer
        self.config = config or ChunkConfig()
        self.text_splitter = self._make_text_splitter(self.config.chunk_size)

    def split(self, text, document_id, metadata):
        chunks = []
        groups = groupby(self._sections(text), key=self._heading_path)
        for number, (path, fragments) in enumerate(groups):
            headings = path or [str(metadata.get("title", ""))]
            prefix = " → ".join(headings) + "\n\n" if self.config.include_headings else ""
            context = dict(metadata, document_id=document_id,
                           section_id=self._id(document_id, str(number)), heading_path=headings)
            bodies = [
                (body, fragment.metadata.get("Code"))
                for fragment in fragments
                for body in self._split_section(fragment.page_content, prefix)
                if body.strip()
            ]
            for part, (body, language) in enumerate(bodies):
                chunk = self._make_chunk(prefix + body, context, part, len(bodies), language)
                chunk["images"] = self._images(body, metadata)
                chunks.append(chunk)
        for index, chunk in enumerate(chunks):
            chunk["previous_chunk_id"] = chunks[index - 1]["id"] if index else None
            chunk["next_chunk_id"] = chunks[index + 1]["id"] if index + 1 < len(chunks) else None
        return chunks

    def _sections(self, text):
        if self.config.splitter == "recursive":
            return [Document(page_content=text)]
        return ExperimentalMarkdownSyntaxTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=True,
        ).split_text(text)

    @staticmethod
    def _heading_path(document):
        return [document.metadata[key] for key in ("h1", "h2", "h3") if key in document.metadata]

    def _make_chunk(self, text, metadata, part, total, language):
        return {
            "id": self._id(metadata["document_id"], metadata["section_id"], str(part), text),
            **{key: metadata[key] for key in ("document_id", "section_id", "heading_path")},
            "code_language": language,
            "section_part": part,
            "section_parts": total,
            "text": text,
            "token_count": self.tokenizer.count(text),
            "source_url": metadata.get("source_url", ""),
            "source_path": metadata.get("source_path"),
            "source_commit": metadata.get("source_commit"),
            "links": metadata.get("links", []),
        }

    def _split_section(self, text: str, prefix: str) -> list[str]:
        budget = self.config.chunk_size - self.tokenizer.count(prefix)
        if budget <= 0:
            raise ValueError("Heading path exceeds chunk size; disable headings or increase chunk_size")
        splitter = self._make_text_splitter(budget) if prefix else self.text_splitter
        return splitter.split_text(text)

    def _make_text_splitter(self, budget):
        return RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=min(self.config.overlap, budget - 1),
            length_function=self.tokenizer.count,
            strip_whitespace=False,
        )

    @staticmethod
    def _id(*parts):
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]

    @staticmethod
    def _images(text, metadata):
        return [image for image in metadata.get("images", []) if any(
            value and value in text
            for value in (image.get("original_url"), image.get("path"), image.get("alt"))
        )]
