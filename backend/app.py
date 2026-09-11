import os
import uuid

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

limiter = Limiter(key_func=get_remote_address)
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
# ВАЖНО — осознанный выбор без компромиссов через .env: Secure всегда
# включён (см. secure=True ниже), даже для локальной разработки. Это
# значит, что ПРОСТОЙ ЗАПУСК ПО http://localhost:8000 (без HTTPS) НЕ БУДЕТ
# ЗАПОМИНАТЬ session_id между сообщениями — браузер не сохранит cookie с
# флагом Secure на незашифрованном соединении, и каждое сообщение будет
# создавать новую анкету. Для локальной разработки нужен HTTPS даже на
# своём компьютере (самоподписанный сертификат) — см. README, раздел
# "Локальный запуск с HTTPS", там же — почему выбран именно этот вариант,
# а не Secure только на проде: cookie в открытом HTTP-трафике можно
# перехватить в той же сети (Wi-Fi, роутер), и тогда чужой сможет
# продолжить диалог от имени кандидата — тот же риск, который мы и
# закрывали, просто через перехват трафика вместо угадывания UUID.
#
# Тело запроса (ChatRequest.session_id ниже) сохранено для ОБРАТНОЙ
# СОВМЕСТИМОСТИ формата ответа (см. ChatResponse), но сервер его больше не
# читает как источник правды — используется только cookie, а если её ещё
# нет (первое обращение), сервер создаёт новую и сразу отдаёт её в ответе.
SESSION_COOKIE_NAME = "session_id"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # год — анкету можно дозаполнить и через полгода

# --- SameSite и Secure: теперь настраиваются, а не зашиты в код -----------
#
# SameSite=Lax (значение по умолчанию) правильно ровно до тех пор, пока чат
# живёт на том же домене, что и сервер. Если виджет встроить на сайт компании,
# а бота держать на отдельном домене или поддомене, запрос к /chat становится
# cross-site, браузер cookie с Lax к нему НЕ приложит, и каждое сообщение
# начнёт новую сессию: диалог не помнится, анкета начинается заново.
#
# Для такого размещения нужен SameSite=None, и тогда браузер ТРЕБУЕТ Secure —
# поэтому ниже он включается принудительно, что бы ни стояло в .env.
#
# SESSION_COOKIE_SECURE=false существует ровно для одного случая: локальная
# проверка по http://localhost без сертификата (в том числе docker compose up,
# который поднимает сайт именно по http). В любом сетевом развёртывании
# оставляйте true: cookie в открытом HTTP-трафике перехватывается в той же
# сети, и чужой человек продолжит диалог от имени кандидата.
SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "lax").strip().lower()
if SESSION_COOKIE_SAMESITE not in ("lax", "strict", "none"):
    print(
        f"[app] SESSION_COOKIE_SAMESITE={SESSION_COOKIE_SAMESITE!r} — недопустимое "
        f"значение (ожидается lax, strict или none). Использую lax."
    )
    SESSION_COOKIE_SAMESITE = "lax"

SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "true").strip().lower() not in ("0", "false", "no")

if SESSION_COOKIE_SAMESITE == "none" and not SESSION_COOKIE_SECURE:
    print("[app] SameSite=None требует Secure — включаю Secure принудительно.")
    SESSION_COOKIE_SECURE = True

if not SESSION_COOKIE_SECURE:
    print(
        "[app] ВНИМАНИЕ: cookie сессии выдаётся БЕЗ флага Secure "
        "(SESSION_COOKIE_SECURE=false). Это допустимо только для локальной "
        "проверки по http://localhost. Для любого сетевого развёртывания "
        "верните true и поднимайте сайт по HTTPS."
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
    answer: str
    context: list
    session_id: str


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

    return ChatResponse(answer=answer, context=list(similar_items), session_id=session_id)


@app.get("/health")
def health():
    return {"status": "ok"}
