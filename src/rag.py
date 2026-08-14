"""Retrieval logic: chunk the story and find passages relevant to a question."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_story(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def chunk_text(text: str, min_len: int = 40) -> list[str]:
    """Split the story into paragraph-sized chunks.

    Paragraphs are separated by blank lines in the source file. Short
    fragments (e.g. a lone title line) are dropped so the index doesn't
    retrieve near-empty chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if len(p) >= min_len]


@dataclass
class RetrievedChunk:
    text: str
    score: float


class RAGIndex:
    """TF-IDF based similarity search over story chunks."""

    def __init__(self, chunks: list[str]):
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(chunks)

    @classmethod
    def from_file(cls, path: str | Path) -> "RAGIndex":
        return cls(chunk_text(load_story(path)))

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            RetrievedChunk(text=self.chunks[i], score=float(scores[i]))
            for i in ranked[:top_k]
        ]
