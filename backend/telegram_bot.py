import asyncio
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

from .assistant import get_answer
from . import rate_limiting

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

    # Та же балльная система по паттернам поведения, что и у сайта (см.
    # backend/rate_limiting.py, backend/app.py) — раньше у Telegram-канала
    # не было вообще никакого ограничения, хотя у сайта уже был лимит
    # slowapi. Вызовы ниже — быстрые операции над словарём в памяти (без
    # сети, без sleep — подтверждено замерами при разработке модуля), не
    # оборачиваются в asyncio.to_thread(), в отличие от get_answer() ниже.
    is_blocked_now, _ = rate_limiting.is_blocked("telegram", chat_id)
    if is_blocked_now:
        await update.message.reply_text(
            "Вы отправляете сообщения слишком часто — пожалуйста, "
            "подождите немного и повторите."
        )
        return
    rate_limiting.record_message("telegram", chat_id, user_message)

    try:
        # get_answer() — обычная (синхронная) функция: расшифровка анкет,
        # поиск похожего вопроса, HTTP-запрос к GigaChat — всё это может
        # занять заметное время (секунды). Без asyncio.to_thread() ниже это
        # выполнялось бы ПРЯМО внутри событийного цикла бота — пока идёт
        # ответ одному кандидату, бот не смог бы начать обрабатывать
        # сообщения от других кандидатов, даже если те уже пришли (весь бот
        # эффективно "замирал" на время одного ответа). to_thread()
        # выполняет get_answer() в отдельном потоке, событийный цикл
        # остаётся свободным — несколько кандидатов обслуживаются
        # параллельно, как и должно быть у изначально асинхронного бота.
        #
        # Внутренняя блокировка candidates.py (threading.Lock, не
        # asyncio.Lock) продолжает защищать сам файл candidates.csv от
        # одновременной записи из разных потоков — это не меняется и не
        # мешает потокам работать параллельно везде, где нет записи в CSV
        # (сама генерация ответа, поиск по базе, обращение к GigaChat).
        answer, _ = await asyncio.to_thread(
            get_answer, user_message, "telegram", chat_id
        )
    except Exception as e:
        # Сюда попадают только сбои ДО обращения к GigaChat (например, не
        # прочитался файл базы знаний) — сама ошибка GigaChat уже обработана
        # без единого технического слова внутри get_answer() (см. assistant.py).
        # Кандидат по-прежнему не должен видеть, что что-то сломалось.
        print(f"[telegram_bot] Непредвиденная ошибка до обращения к GigaChat: {e}")
        answer = (
            "Пока не могу сформулировать точный ответ на этот вопрос. "
            "Уточните, пожалуйста, что именно вас интересует, либо "
            "свяжитесь с менеджером напрямую."
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