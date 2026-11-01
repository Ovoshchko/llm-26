from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=6000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank")
        return value


class Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    query: str = Field(min_length=1, max_length=1000)

    @field_validator("query")
    @classmethod
    def valid_query(cls, value):
        value = value.strip()
        if value.casefold() in {"", "null", "none"}:
            raise ValueError("A requirement needs a meaningful search query")
        return value


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirements: list[Requirement] = Field(min_length=1, max_length=4)


class EvidenceQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: PositiveInt
    source_id: PositiveInt
    quote: str = Field(min_length=10, max_length=2000)


class SearchStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: list[EvidenceQuote] = Field(max_length=8)
    next_requirement_id: PositiveInt | None
    next_query: str | None = Field(min_length=1, max_length=1000)

    @field_validator("next_query", mode="before")
    @classmethod
    def normalize_next_query(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if value.casefold() in {"", "null", "none"}:
                return None
        return value


class EvidenceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str | None = Field(max_length=1000)

    requirement_id: PositiveInt
    status: Literal["supported", "partial", "unsupported"]
    source_ids: list[PositiveInt] = Field(max_length=8)
    reason: str = Field(min_length=1, max_length=1500)
    missing_information: str | None = Field(max_length=1000)
    next_query: str | None = Field(max_length=1000)

    normalize_next_query = field_validator("next_query", mode="before")(SearchStep.normalize_next_query.__func__)


class DraftAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    source_ids: list[PositiveInt] = Field(max_length=20)


class Citation(BaseModel):
    id: int
    chunk_id: str
    source_url: str
    source_path: str
    section: str
    page: int | None = None
    snippet: str
    score: float


class RagAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    insufficient_context: bool
    metadata: dict
