import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

from .assistant import get_answer

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Please set it in your .env file.")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я консультант компании. Задайте вопрос — отвечу на основе базы знаний."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_id = str(update.effective_chat.id)

    try:
        answer, _ = get_answer(user_message, source="telegram", external_id=chat_id)
    except Exception as e:
        print(f"[telegram_bot] Ошибка: {e}")
        answer = (
            "Сейчас нет доступа к серверу ассистента по техническим причинам. "
            "Пожалуйста, напишите чуть позже."
        )

    await update.message.reply_text(answer)


def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Telegram bot started. Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == "__main__":
    main()