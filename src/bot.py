"""Telegram bot: answers questions about data/story.txt using RAG + Claude."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from src.llm import generate_answer
from src.rag import RAGIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STORY_PATH = Path(__file__).resolve().parent.parent / "data" / "story.txt"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я отвечаю на вопросы по рассказу «Маяк на мысе Туманном». "
        "Просто напиши свой вопрос."
    )


async def answer_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    index: RAGIndex = context.bot_data["index"]
    question = update.message.text
    chunks = index.retrieve(question, top_k=3)
    answer = generate_answer(question, chunks)
    await update.message.reply_text(answer)


def build_app():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = ApplicationBuilder().token(token).build()
    app.bot_data["index"] = RAGIndex.from_file(STORY_PATH)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_question))
    return app


def main() -> None:
    app = build_app()
    logger.info("Bot started, polling for updates...")
    app.run_polling()


if __name__ == "__main__":
    main()
