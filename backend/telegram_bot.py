import asyncio
import os

from dotenv import load_dotenv
from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .assistant import get_answer
from . import anketa
from . import candidates
from . import notifications
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

# Постоянная кнопка под полем ввода. Нажатие отправляет обычное текстовое
# сообщение «Заполнить анкету», которое код распознаёт напрямую
# (anketa.is_anketa_request) — нейросеть при этом не вызывается вообще.
#
# Зачем кнопка. Без неё запуск анкеты зависел от того, поставит ли модель
# служебный маркер. В боевом тесте кандидат написал «запонляем анкетиу» с
# опечаткой и застрял: маркер был отклонён защитой, анкета не началась.
# Кнопка убирает этот путь целиком — человеку не нужно угадывать формулировку.
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [[anketa.ANKETA_BUTTON_TEXT]],
    resize_keyboard=True,
    is_persistent=True,
    input_field_placeholder="Спросите о работе или нажмите кнопку",
)


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


async def reply_long(update: Update, text: str, options: list | None = None) -> None:
    """reply_text с учётом лимита длины: отправляет ответ одним или
    несколькими сообщениями подряд.

    options — варианты ответа на текущий вопрос анкеты. Показываются
    клавиатурой под полем ввода: на телефоне нажать кнопку заметно проще,
    чем набирать «Промышленный альпинист», а в таблицу попадает ровно одно
    из допустимых значений, а не десяток вариантов написания одного и того
    же. Клавиатура ставится только на ПОСЛЕДНЕЕ сообщение — если ответ
    разбит на части, кнопки должны быть внизу, под вопросом.

    Когда вариантов нет, ставится постоянная кнопка «Заполнить анкету».
    Клавиатура меняется на каждом сообщении намеренно: иначе кнопки от
    предыдущего вопроса анкеты остались бы висеть на следующем, где нужен
    свободный ввод."""
    parts = split_message(text)
    for part in parts[:-1]:
        await update.message.reply_text(part)

    if options:
        # Идёт анкета: под полем ввода варианты ответа на текущий вопрос.
        # one_time_keyboard — клавиатура прячется сразу после нажатия,
        # resize_keyboard — кнопки по высоте текста, а не в пол-экрана.
        markup = ReplyKeyboardMarkup(
            [[option] for option in options],
            one_time_keyboard=True,
            resize_keyboard=True,
        )
    else:
        # Обычный разговор: постоянная кнопка «Заполнить анкету».
        markup = MAIN_KEYBOARD

    await update.message.reply_text(parts[-1], reply_markup=markup)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Здравствуйте! Я помогу разобраться с работой в ПОЛАТИ: вакансии, "
        "зарплата, вахта, обучение, документы. Спрашивайте — отвечу по делу.\n\n"
        "Если готовы оставить заявку — нажмите кнопку внизу.",
        reply_markup=MAIN_KEYBOARD,
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
    # Во время анкеты счётчик не работает — см. rate_limiting.in_anketa.
    in_anketa = await asyncio.to_thread(rate_limiting.in_anketa, "telegram", chat_id)

    is_blocked_now, _ = (False, 0.0) if in_anketa else rate_limiting.is_blocked("telegram", chat_id)
    if is_blocked_now:
        await update.message.reply_text(
            "Вы отправляете сообщения слишком часто — пожалуйста, "
            "подождите немного и повторите."
        )
        return
    if not in_anketa:
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

    # Варианты запрашиваются ПОСЛЕ get_answer: анкета уже перешла к
    # следующему полю, значит кнопки будут к тому вопросу, который кандидат
    # видит в этом же сообщении.
    candidate_id = await asyncio.to_thread(candidates.find_candidate, "telegram", chat_id)
    options = await asyncio.to_thread(anketa.pending_field_options, candidate_id)

    await reply_long(update, answer, options)


async def handle_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ответ на голосовое, фото, документ, стикер и прочее не-текстовое.

    Раньше фильтр filters.TEXT просто отбрасывал такие сообщения, и бот
    молчал. Человек присылал голосовое с вопросом или фото паспорта, не
    получал НИЧЕГО в ответ и решал, что бот сломался, — после чего уходил.
    Голосовыми в Telegram пользуются очень многие.

    Распознавать речь мы не беремся, но сказать об этом обязаны."""
    await update.message.reply_text(
        "Я понимаю только текстовые сообщения — голосовые, фото и файлы, "
        "к сожалению, прочитать не могу. Напишите, пожалуйста, текстом.\n\n"
        "Документы присылать боту не нужно: всё необходимое вы предоставите "
        "менеджеру и в отделе кадров."
    )


async def show_card(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Нажатие кнопки «Показать все данные кандидата» в HR-группе.

    Проверка чата обязательна. callback_data приходит от Telegram и содержит
    id кандидата — без проверки любой человек, узнавший формат строки, мог бы
    вытащить чужую анкету из личной переписки с ботом. Отвечаем полной
    карточкой только если кнопку нажали именно в той группе, которая указана
    в TELEGRAM_HR_GROUP_CHAT_ID."""
    query = update.callback_query
    await query.answer()

    _, hr_chat_id = notifications._settings()
    if not hr_chat_id or str(query.message.chat.id) != str(hr_chat_id):
        await query.edit_message_reply_markup(reply_markup=None)
        return

    data = query.data or ""
    if not data.startswith("card:"):
        return
    candidate_id = data.split(":", 1)[1]

    card = await asyncio.to_thread(candidates.get_card, candidate_id)
    if not card:
        await query.message.reply_text("Карточка не найдена — возможно, кандидат удалил свои данные.")
        return

    text = notifications.format_full_card(card)
    for part in split_message(text):
        await query.message.reply_text(part, parse_mode="HTML")

    # Кнопку убираем: данные уже показаны, повторные нажатия только
    # засоряли бы группу копиями одной и той же анкеты.
    await query.edit_message_reply_markup(reply_markup=None)


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
    # Нажатия кнопок под уведомлениями. Фильтра по типу чата здесь нет
    # намеренно: кнопка живёт в HR-группе, а проверку, что нажали именно
    # там, делает сам show_card.
    application.add_handler(CallbackQueryHandler(show_card, pattern=r"^card:"))
    # Всё остальное в личных сообщениях: голос, фото, документы, стикеры.
    # Регистрируется ПОСЛЕ текстового обработчика — Telegram отдаёт
    # сообщение первому подходящему, и текст успевает перехватить тот.
    application.add_handler(
        MessageHandler(
            ~filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
            handle_non_text,
        )
    )

    print("Telegram bot started. Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == "__main__":
    main()