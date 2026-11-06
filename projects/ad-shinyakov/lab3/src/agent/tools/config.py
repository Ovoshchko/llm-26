from dataclasses import dataclass
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[3]
LAB2_ROOT = LAB_ROOT.parent / 'lab2'


@dataclass(frozen=True)
class ToolConfig:
    embeddings: Path = LAB2_ROOT / 'data/embeddings/ced08184c9da0803'
    qdrant_url: str = 'http://localhost:6333'
    request_timeout: int = 10
