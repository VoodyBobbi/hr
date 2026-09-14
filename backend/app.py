import os
import uuid
import warnings

from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .assistant import get_answer
from . import anketa
from . import candidates
from . import rate_limiting

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI()

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def read_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# --- CORS: разрешаем обращаться к /chat из браузера только с сайтов,
# перечисленных в ALLOWED_ORIGINS (.env) — не с ЛЮБОГО сайта в интернете.
#
# ВАЖНО (чтобы не путать это с полноценной защитой от нагрузки): CORS —
# это правило для БРАУЗЕРА, а не для сервера. Он не даёт постороннему сайту
# незаметно встроить этот чат в свою страницу через JavaScript в браузере
# случайного посетителя. Он НЕ мешает прямому запросу из Postman, curl или
# любого скрипта — у них нет браузера, который проверяет CORS. От прямых
# скриптов и потока запросов защищает ограничение частоты запросов ниже
# (Limiter), а не эта настройка.
#
# allow_credentials=True (ниже) обязателен теперь, что и раньше не было
# нужно: SESSION_COOKIE_NAME ниже — cookie, а браузер по умолчанию НЕ
# прикладывает cookie к cross-origin fetch()-запросам без явного
# credentials: "include" на клиенте (см. frontend/index.html) И
# allow_credentials: true на CORS-мидлваре сервера — без обеих сторон
# сразу session_id-cookie просто не доедет до сервера.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]

if not ALLOWED_ORIGINS:
    print(
        "[app] ВНИМАНИЕ: ALLOWED_ORIGINS не задан в .env — CORS не разрешит "
        "обращаться к /chat ни с одного сайта в браузере (кроме прямых "
        "запросов не из браузера). Добавьте свой домен в .env, если чат "
        "должен работать на сайте."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


# --- Ограничение частоты запросов к /chat: не больше CHAT_RATE_LIMIT_PER_MINUTE
# сообщений в минуту с одного IP-адреса. Без этого один пользователь или
# программа мог(ла) отправлять запросы без остановки, и это била бы либо по
# счёту за GigaChat, либо по стабильности сервера (см. README).
CHAT_RATE_LIMIT_PER_MINUTE = int(os.getenv("CHAT_RATE_LIMIT_PER_MINUTE", "20"))

# config_filename здесь ОБЯЗАТЕЛЕН, хотя выглядит странно.
#
# slowapi при создании Limiter сам, без спроса, ищет в текущей папке файл
# .env и читает его через starlette.config.Config. А тот открывает файл БЕЗ
# указания кодировки — то есть в кодировке, принятой в системе. На русской
# Windows это cp1251, и любая кириллица в .env валит запуск с ошибкой:
#
#   UnicodeDecodeError: 'charmap' codec can't decode byte 0x98
#
# Падает при этом ИМПОРТ backend.app, поэтому сайт не поднимается вообще.
# В консоли это выглядит особенно обманчиво: Telegram-бот стартует как ни в
# чём не бывало, строка про запущенный сайт печатается, а самого сайта нет.
#
# Своё содержимое проекта читает сам, через load_dotenv (python-dotenv
# открывает файл в UTF-8 явно и таких проблем не имеет). Ограничителю
# частоты запросов наш .env не нужен вообще — просто уводим его на
# несуществующий путь, и файл он не трогает.
#
# catch_warnings глушит предупреждение starlette о ненайденном файле —
# файла и не должно быть, сообщать тут не о чем.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    limiter = Limiter(
        key_func=get_remote_address,
        config_filename="slowapi-config-unused",
    )
app.state.limiter = limiter


# --- session_id: раньше клиент присылал его в теле запроса, и сервер
# доверял ЛЮБОМУ присланному значению — подставив чужой session_id (угадав
# или подсмотрев его), можно было получить доступ к чужому диалогу и чужой
# анкете (не требовало ничего, кроме умения отправить POST-запрос с нужным
# телом — curl, Postman, что угодно).
#
# Теперь сервер сам генерирует и выдаёт session_id при первом обращении —
# через HttpOnly + Secure cookie (JavaScript её прочитать/подделать не
# может, в отличие от localStorage, где значение раньше лежало открытым и
# было видно через DevTools; Secure не даёт браузеру передать её по
# незашифрованному HTTP — только по HTTPS). Браузер дальше прикладывает эту
# cookie автоматически на каждый следующий запрос — фронтенду
# (frontend/index.html) для этого ничего специально делать не нужно, кроме
# credentials: "include" в fetch().
#
# Флаг Secure НЕ зашит жёстко: он выводится из ALLOWED_ORIGINS — см. блок
# «Secure и SameSite у cookie сессии» чуть ниже. На локальном http он
# выключается, на сетевом адресе включается сам.
#
# Раньше он был включён всегда, и здесь стояло длинное объяснение, почему
# так правильно, со ссылкой на раздел README «Локальный запуск с HTTPS».
# Ни того поведения, ни того раздела больше нет — текст пережил правку и
# вводил в заблуждение любого, кто открывал файл.
#
# Зачем Secure нужен там, где он включается: cookie в открытом HTTP-трафике
# перехватывается в той же сети (чужой Wi-Fi, роутер), и тогда посторонний
# продолжит диалог от имени кандидата. Это тот же риск, ради которого
# session_id вообще убрали из тела запроса, просто через перехват трафика
# вместо угадывания.
#
# Тело запроса (ChatRequest.session_id ниже) сохранено для ОБРАТНОЙ
# СОВМЕСТИМОСТИ формата ответа (см. ChatResponse), но сервер его больше не
# читает как источник правды — используется только cookie, а если её ещё
# нет (первое обращение), сервер создаёт новую и сразу отдаёт её в ответе.
SESSION_COOKIE_NAME = "session_id"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # год — анкету можно дозаполнить и через полгода

# --- Secure и SameSite у cookie сессии ------------------------------------
#
# Обе настройки ВЫВОДЯТСЯ САМИ из ALLOWED_ORIGINS, потому что там уже
# записано всё нужное: по какому адресу открывается сайт и на том же он
# домене, что сервер, или на другом. Отдельных переменных в .env для этого
# больше нет — они только множили способы ошибиться, а ошибка в них
# проявлялась неочевидно: бот переставал помнить диалог, и понять почему
# было невозможно.
#
# Secure означает «отправлять cookie только по HTTPS». Если его включить на
# сайте, открытом по обычному http://localhost, браузер cookie просто не
# сохранит — каждое сообщение станет новой сессией, и анкета будет
# начинаться заново.
#
# Поэтому правило простое: все адреса локальные и по http — Secure
# выключаем, иначе включаем. На сервере с настоящим доменом он включится
# сам, забыть про это невозможно.
def _looks_local(origin: str) -> bool:
    origin = origin.strip().lower()
    if origin.startswith("https://"):
        return False
    return "localhost" in origin or "127.0.0.1" in origin or "0.0.0.0" in origin


_all_local = bool(ALLOWED_ORIGINS) and all(_looks_local(o) for o in ALLOWED_ORIGINS)

# SameSite=None нужен, только если чат встроен на сайт с ДРУГИМ доменом,
# чем сервер бота: браузер иначе не приложит cookie к такому запросу.
# Определяем по тому, есть ли среди разрешённых адресов внешний домен.
_has_external = any(not _looks_local(o) for o in ALLOWED_ORIGINS)

SESSION_COOKIE_SECURE = not _all_local
SESSION_COOKIE_SAMESITE = "none" if (_has_external and SESSION_COOKIE_SECURE) else "lax"

# Переменные окружения всё ещё имеют приоритет — как аварийный рычаг для
# нестандартного развёртывания (например, HTTPS обеспечивает внешний прокси,
# а сам контейнер об этом не знает). В .env.example они намеренно не
# описаны: в обычной работе трогать их не нужно.
if os.getenv("SESSION_COOKIE_SECURE"):
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE").strip().lower() not in ("0", "false", "no")
if os.getenv("SESSION_COOKIE_SAMESITE"):
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE").strip().lower()
    if SESSION_COOKIE_SAMESITE not in ("lax", "strict", "none"):
        SESSION_COOKIE_SAMESITE = "lax"

# Браузер отвергает SameSite=None без Secure — чиним молча, иначе cookie
# не сохранится вообще ни при каких условиях.
if SESSION_COOKIE_SAMESITE == "none" and not SESSION_COOKIE_SECURE:
    SESSION_COOKIE_SECURE = True

print(
    f"[app] Cookie сессии: Secure={'вкл' if SESSION_COOKIE_SECURE else 'выкл'}, "
    f"SameSite={SESSION_COOKIE_SAMESITE} "
    f"({'локальный запуск по http' if _all_local else 'сетевое развёртывание'}, "
    f"определено по ALLOWED_ORIGINS)."
)


def _set_session_cookie(response: Response, session_id: str) -> None:
    """Одно место выдачи cookie — раньше set_cookie вызывался в двух ветках
    /chat с продублированными параметрами, и любая правка требовала не забыть
    про обе."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite=SESSION_COOKIE_SAMESITE,
        secure=SESSION_COOKIE_SECURE,
    )


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    # Намеренно НЕ используем стандартный обработчик slowapi как есть: он по
    # умолчанию отдаёт текст вроде "Rate limit exceeded: 20 per 1 minute" —
    # это техническая формулировка, а кандидат не должен видеть ничего, что
    # звучит как ошибка системы (см. README про то, что клиент не должен
    # видеть технические сообщения). Ответ ниже отдаётся в ТОМ ЖЕ формате
    # {answer, context, session_id}, что и обычный /chat, чтобы фронтенду не
    # нужно было отдельно обрабатывать код 429 — сообщение просто придёт
    # как обычный ответ бота.
    session_id = request.cookies.get(SESSION_COOKIE_NAME) or str(uuid.uuid4())
    return JSONResponse(
        status_code=200,
        content={
            "answer": (
                "Вы отправляете сообщения слишком часто — пожалуйста, "
                "подождите немного и повторите."
            ),
            "context": [],
            "session_id": session_id,
        },
    )


# top_k: сколько похожих вопросов искать в базе FAQ на каждое сообщение.
# Раньше клиент присылал это значение в теле запроса — можно было передать
# top_k=1000, заставляя FAISS искать в 300+ раз больше, чем нужно, на
# каждое сообщение (реальная нагрузка на сервер, управляемая снаружи, не
# самим сервером). Ограничение диапазона (было: Field(ge=1, le=10)) снижало
# риск, но не убирало саму возможность клиента влиять на этот параметр —
# теперь его вообще нет в ChatRequest, значение фиксировано константой.
TOP_K = 3


class ChatRequest(BaseModel):
    # max_length ограничивает одно сообщение — не количество запросов в минуту
    # (это делает rate limit выше), а размер ОДНОГО сообщения. Без этого
    # ничего не мешает прислать один запрос на 100 000 символов — это тоже
    # расходует токены GigaChat и замедляет ответ, даже если запросов мало.
    # 2000 символов — с запасом на длинный вопрос от кандидата, но отсекает
    # явно неразумный ввод (вставленный документ целиком, спам-атаку).
    message: str = Field(..., min_length=1, max_length=2000)
    # Поле сохранено для совместимости формата запроса — сервер его больше
    # не читает как источник session_id (см. SESSION_COOKIE_NAME выше).
    session_id: str | None = None


class ChatResponse(BaseModel):
    # options — варианты ответа для текущего вопроса анкеты. Фронтенд
    # показывает их кнопками вместо поля ввода (см. frontend/index.html).
    # Пустой список означает свободный ввод.
    answer: str
    context: list
    session_id: str
    options: list[str] = []


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{CHAT_RATE_LIMIT_PER_MINUTE}/minute")
def chat(request: Request, response: Response, body: ChatRequest,
         session_id: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)):
    is_new_session = session_id is None
    if is_new_session:
        session_id = str(uuid.uuid4())

    # Балльная система по паттернам поведения (backend/rate_limiting.py) —
    # дополняет лимит slowapi выше (тот просто считает число запросов в
    # минуту с IP, не заботясь об их содержании). Проверяется ДО get_answer,
    # чтобы заблокированный кандидат не тратил впустую вызов GigaChat.
    #
    # Балльный счётчик привязан к СЕССИИ браузера, а не к IP-адресу.
    #
    # По IP считать нельзя: за одним внешним адресом сидят сотня коллег в
    # офисе, посетители кафе с общим Wi-Fi или целый район на NAT мобильного
    # оператора. Их сообщения сливаются в один поток, промежутки между ними
    # оказываются короткими, и счётчик срабатывал бы от обычной работы
    # десятка людей — блокируя всех разом за то, чего никто не делал.
    #
    # Обход через очистку cookie при этом остаётся возможен, и это
    # осознанный размен: от него защищает лимит slowapi выше — не больше
    # CHAT_RATE_LIMIT_PER_MINUTE запросов в минуту с одного IP, независимо
    # от cookie. Он ограничивает ущерб, но, в отличие от блокировки, не
    # выключает бота для целого офиса. Если поток с одного адреса всё же
    # мешает, уменьшайте CHAT_RATE_LIMIT_PER_MINUTE в .env. Если сайт стоит
    # за обратным прокси, настройте передачу X-Forwarded-For и запускайте
    # uvicorn с --proxy-headers, иначе для slowapi все посетители будут
    # выглядеть одним адресом.
    session_key = f"session:{session_id}"

    # Во время анкеты счётчик не работает вообще — см. rate_limiting.in_anketa.
    in_anketa = rate_limiting.in_anketa("site", session_id)

    is_blocked_now, _ = (False, 0.0) if in_anketa else rate_limiting.is_blocked("site", session_key)
    if is_blocked_now:
        if is_new_session:
            _set_session_cookie(response, session_id)
        return ChatResponse(
            answer=(
                "Вы отправляете сообщения слишком часто — пожалуйста, "
                "подождите немного и повторите."
            ),
            context=[],
            session_id=session_id,
        )
    if not in_anketa:
        rate_limiting.record_message("site", session_key, body.message)

    answer, similar_items = get_answer(
        body.message,
        source="site",
        external_id=session_id,
        top_k=TOP_K,
    )

    if is_new_session:
        _set_session_cookie(response, session_id)

    # Варианты запрашиваются ПОСЛЕ get_answer: к этому моменту анкета уже
    # перешла к следующему полю, и мы получаем кнопки именно к тому вопросу,
    # который кандидат видит в ответе выше.
    options = anketa.pending_field_options(candidates.find_candidate("site", session_id))

    return ChatResponse(
        answer=answer,
        context=list(similar_items),
        session_id=session_id,
        options=options,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
