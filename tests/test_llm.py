from src.llm import build_prompt
from src.rag import RetrievedChunk


def test_build_prompt_includes_question_and_chunks():
    chunks = [
        RetrievedChunk(text="Первый отрывок текста.", score=0.9),
        RetrievedChunk(text="Второй отрывок текста.", score=0.5),
    ]
    prompt = build_prompt("Кто главный герой?", chunks)

    assert "Кто главный герой?" in prompt
    assert "Первый отрывок текста." in prompt
    assert "Второй отрывок текста." in prompt


def test_build_prompt_numbers_chunks_in_order():
    chunks = [
        RetrievedChunk(text="A", score=1.0),
        RetrievedChunk(text="B", score=0.8),
    ]
    prompt = build_prompt("q", chunks)

    assert prompt.index("[Отрывок 1]") < prompt.index("[Отрывок 2]")
    assert prompt.index("A") < prompt.index("[Отрывок 2]")
