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

# Жёсткий предел Telegram Bot API на одно текстовое сообщение. Отправка
# длиннее приводит к ошибке "Message is too long", и кандидат не получает
# ответ ВООБЩЕ. Реально в этот предел упираются готовая карточка анкеты из
# 29 полей (anketa.format_card_for_confirmation) и длинные ответы GigaChat.
TELEGRAM_MESSAGE_LIMIT = 4096


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list:
    """Режет длинный ответ на части, влезающие в одно сообщение Telegram.

    Режет по границам: сначала пробует разделить по пустой строке, затем по
    переводу строки, и только если и это не помогло (одна строка длиннее
    лимита) — по живому. Это важно для карточки анкеты: разрыв посередине
    строки "СНИЛС: 123..." выглядит как потерянные данные."""
    if len(text) <= limit:
        return [text]

    parts = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts


async def reply_long(update: Update, text: str) -> None:
    """reply_text с учётом лимита длины: отправляет ответ одним или
    несколькими сообщениями подряд."""
    for part in split_message(text):
        await update.message.reply_text(part)


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

    await reply_long(update, answer)


def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # filters.ChatType.PRIVATE — только личные сообщения кандидатов.
    #
    # Без этого фильтра бот отвечал в ЛЮБОМ чате, куда его добавили, включая
    # HR-группу для уведомлений о новых заявках (backend/notifications.py).
    # Каждая реплика рекрутёров в этой группе воспринималась как сообщение
    # кандидата: бот отвечал на неё в группе и заводил "анкету" на chat_id
    # самой группы вместо живого человека.
    #
    # Тот же фильтр стоит и на /start: команда в группе больше не вызывает
    # приветствие, адресованное кандидату.
    application.add_handler(
        CommandHandler("start", start_command, filters=filters.ChatType.PRIVATE)
    )
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
            handle_message,
        )
    )

    print("Telegram bot started. Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == "__main__":
    main()