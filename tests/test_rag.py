from pathlib import Path

import pytest

from src.rag import RAGIndex, chunk_text, load_story

STORY_PATH = Path(__file__).resolve().parent.parent / "data" / "story.txt"


def test_load_story_reads_file():
    text = load_story(STORY_PATH)
    assert "Егор Найдёнов" in text


def test_chunk_text_splits_on_blank_lines():
    text = (
        "Первый абзац с достаточно длинным текстом, чтобы пройти фильтр.\n\n"
        "Второй абзац тоже вполне себе длинный и содержательный текст.\n\n"
        "x"
    )
    chunks = chunk_text(text)
    assert len(chunks) == 2
    assert "Первый абзац" in chunks[0]
    assert "Второй абзац" in chunks[1]


def test_chunk_text_drops_short_fragments():
    text = "Заголовок\n\n" + "Настоящий длинный абзац с содержанием истории. " * 3
    chunks = chunk_text(text)
    assert all(len(c) >= 40 for c in chunks)
    assert not any(c == "Заголовок" for c in chunks)


def test_rag_index_rejects_empty_chunks():
    with pytest.raises(ValueError):
        RAGIndex([])


def test_retrieve_finds_relevant_chunk_about_the_dog():
    index = RAGIndex.from_file(STORY_PATH)
    results = index.retrieve("Как звали собаку смотрителя маяка?", top_k=1)
    assert len(results) == 1
    assert "Компас" in results[0].text


def test_retrieve_finds_relevant_chunk_about_the_children():
    index = RAGIndex.from_file(STORY_PATH)
    results = index.retrieve("Кто были дети, которых унесло в лодке?", top_k=1)
    assert "Тимофей" in results[0].text or "Ася" in results[0].text


def test_retrieve_respects_top_k():
    index = RAGIndex.from_file(STORY_PATH)
    results = index.retrieve("маяк", top_k=2)
    assert len(results) == 2


def test_retrieve_scores_are_sorted_descending():
    index = RAGIndex.from_file(STORY_PATH)
    results = index.retrieve("шторм и рыбацкая лодка", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
