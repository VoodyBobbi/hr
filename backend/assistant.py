import json
import os
import re
import threading
import time
from collections import OrderedDict

import numpy as np
from dotenv import load_dotenv
import gigachat.context
from gigachat import GigaChat
from gigachat.exceptions import (
    AuthenticationError,
    GigaChatException,
    RateLimitError,
    ResponseError,
)
from gigachat.models import Chat, Messages, MessagesRole
from sentence_transformers import SentenceTransformer

from .rag_index import load_index, search_similar
from . import anketa
from . import candidates
from . import crypto_utils
from . import logger
from . import notifications
from . import paths
from .validators import FieldValidationError

load_dotenv()

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")

# Сколько секунд ждать ответа нейросети, прежде чем считать вызов зависшим.
GIGACHAT_TIMEOUT_SECONDS = float(os.getenv("GIGACHAT_TIMEOUT_SECONDS", "30"))
if not GIGACHAT_CREDENTIALS:
    raise RuntimeError("GIGACHAT_CREDENTIALS is not set. Please set it in your .env file.")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH = os.path.join(DATA_DIR, "faqs_metadata.npy")
# Отдельный, второй FAISS-индекс для базы знаний из data/kb/ (см.
# backend/kb_chunker.py, backend/build_index.py) — раздельный от FAQ-индекса
# выше по требованию "две раздельные поиска, не один общий top-3" (см.
# обсуждение пункта про сжатие промпта/базы): вопрос кандидата ищется И в
# faqs.json (готовые ответы, search-first — см. get_answer), И отдельно в
# базы знаний (факты для контекста GigaChat, если FAQ не нашёл
# готового ответа) — это два разных назначения, смешивать их в один общий
# top-3 давало бы, например, что более релевантный FAQ-ответ может быть
# вытеснен менее релевантным KB-фактом просто по случайному порядку.
KB_INDEX_PATH = os.path.join(DATA_DIR, "kb_faiss_index.bin")
KB_META_PATH = os.path.join(DATA_DIR, "kb_metadata.npy")
AGENT_PROMPT_PATH = os.path.join(DATA_DIR, "system_prompt.md")
KB_DIR = os.path.join(DATA_DIR, "kb")

# История переписки — персональные данные кандидата, поэтому лежит ВНЕ
# репозитория (папка history/ в корне проекта, см. backend/paths.py), в
# отличие от базы знаний бота выше (FAQ/промпт — это конфигурация, не ПД).
HISTORY_DIR = paths.HISTORY_DIR

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Пороги близости для поиска ------------------------------------------
#
# Индекс — faiss.IndexFlatIP на нормализованных векторах, поэтому поиск
# возвращает сразу КОСИНУСНУЮ БЛИЗОСТЬ: число от -1 до 1, где больше значит
# похожее. 0.45 — «примерно об одном», 0.75 — «почти то же самое».
#
# Раньше индекс был IndexFlatL2 и возвращал квадрат расстояния (меньше значит
# похожее, диапазон 0..4). Порог в таких единицах невозможно осмыслить на
# глаз, и именно поэтому исходное значение было выставлено неверно.
#
# Пороги разные, потому что у поисков разное назначение:
# FAQ_MIN_SIMILARITY — готовый ответ уходит кандидату НАПРЯМУЮ, без GigaChat,
# поэтому порог строгий: ошибка здесь видна кандидату как уверенный ответ не
# на его вопрос. KB_MIN_SIMILARITY — найденные факты только подмешиваются в
# промпт, GigaChat всё равно решает, что из них использовать, поэтому здесь
# лишний факт дешевле пропущенного.
#
# Оба значения выносятся в .env, чтобы калибровать их по реальным логам без
# правки кода (см. README, раздел «Калибровка порогов поиска»).
FAQ_MIN_SIMILARITY = float(os.getenv("FAQ_MIN_SIMILARITY", "0.62"))
KB_MIN_SIMILARITY = float(os.getenv("KB_MIN_SIMILARITY", "0.35"))

# Сколько фактов из базы знаний максимум уходит в промпт и какой длины.
# Ограничение нужно ради предсказуемости расходов: без него редкий вопрос,
# зацепивший много длинных чанков, раздувал бы промпт в разы. Символы, а не
# токены — точный подсчёт токенов требовал бы токенизатора GigaChat, которого
# у нас нет, а порядок величины отражается верно (для русского текста грубо
# 2-3 символа на токен).
KB_TOP_K = int(os.getenv("KB_TOP_K", "5"))
KB_CONTEXT_CHAR_LIMIT = int(os.getenv("KB_CONTEXT_CHAR_LIMIT", "2500"))

# Файлы, чьё время изменения проверяется на каждом запросе (_ensure_fresh):
# правка любого из них подхватывается на лету, без перезапуска сервера.
# Сама папка kb/ сюда не входит — её содержимое участвует через свой индекс
# (kb_faiss_index.bin), который пересобирается build_index.py; отслеживать
# ещё и исходные json-файлы значило бы дублировать эту проверку.
WATCHED_FILES = [AGENT_PROMPT_PATH, INDEX_PATH, META_PATH, KB_INDEX_PATH, KB_META_PATH]

MAX_HISTORY_MESSAGES = 20

# Модель эмбеддингов загружается ЛЕНИВО — при первом обращении, а не при
# импорте модуля.
#
# Раньше здесь стояло embedding_model = SentenceTransformer(...), и модель
# (около 470 МБ) читалась с диска в память в момент `import backend.assistant`.
# Три следствия, все неприятные:
#
# 1. Приложение не импортировалось, пока модель не загрузится целиком.
#    Для uvicorn это значит, что сервер не поднимается и не отвечает даже на
#    /health — балансировщик на сервере считает такой сервис мёртвым, хотя
#    он просто прогревается.
# 2. Если модели не оказалось в кеше и до huggingface.co нет доступа, падал
#    импорт, то есть всё приложение — из-за части, которая нужна только для
#    поиска.
# 3. При запуске через start.py модель какое-то время жила в памяти в двух
#    экземплярах: свой был у build_index.py, который отрабатывает раньше.
#
# Теперь импорт лёгкий, сервер поднимается сразу, а модель подгружается на
# первом вопросе кандидата (или раньше — при сборке индекса).
_embedding_model = None
_embedding_model_lock = threading.Lock()


def get_embedding_model():
    """Модель эмбеддингов, загруженная по требованию. Повторные вызовы
    возвращают тот же объект.

    Блокировка нужна на случай, когда два первых запроса пришли
    одновременно: без неё оба начали бы грузить модель, и в памяти
    ненадолго оказалось бы две копии по 470 МБ. Проверка внутри блокировки
    повторяется намеренно — второй поток, дождавшийся своей очереди, должен
    увидеть уже готовую модель, а не грузить её заново."""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                print("[assistant] Загружаю модель поиска, это разовая операция...")
                started = time.time()
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                elapsed_ms = int((time.time() - started) * 1000)
                print(f"[assistant] Модель поиска готова за {elapsed_ms / 1000:.1f} с.")
                logger.log_event(
                    logger.EVENT_MODEL,
                    f"Модель поиска загружена ({EMBEDDING_MODEL_NAME}).",
                    duration_ms=elapsed_ms,
                )
    return _embedding_model

_state_lock = threading.RLock()
_state = {
    "mtimes": {},
    "agent_prompt": "",
    "index": None,
    "metadata": None,
}

_history_lock = threading.Lock()


def _get_mtimes() -> dict:
    return {
        path: os.path.getmtime(path) if os.path.exists(path) else None
        for path in WATCHED_FILES
    }


# Слова, по которым видно, что кандидат ДЕЙСТВИТЕЛЬНО согласился начать
# анкету. Проверяются перед тем, как принять маркер START_ANKETA от модели.
_ANKETA_INTENT = (
    "да", "ага", "угу", "давай", "давайте", "хочу", "хочется", "готов", "готова",
    "согласен", "согласна", "начнём", "начнем", "начинаем", "поехали", "оформляй",
    "оформляйте", "оформляемся", "заполнить", "заполняем", "заполню", "анкету",
    "анкета", "заявку", "заявка", "откликнуться", "отклик", "устроиться",
    "ок", "окей", "буду", "валяй", "погнали", "конечно", "yes",
)


def _user_really_asked_for_anketa(user_message: str) -> bool:
    """Правда ли кандидат просил начать анкету.

    Нужна потому, что GigaChat ставит маркер START_ANKETA слишком охотно.
    В боевом тесте кандидат спросил «Какие есть варианты?» — обычный вопрос
    о профессиях — а модель прислала маркер, и человеку вместо ответа
    показали согласие на обработку персональных данных.

    Сравнение идёт по ЦЕЛЫМ СЛОВАМ. Первая версия искала совпадение только
    по началу слова, и это было хуже исходной болезни: «\bда» срабатывало
    внутри «данные», «далеко», «даже», а «\bок» — внутри «окно» и «около».
    Вопрос «расскажите про данные» запускал анкету. Зеркально не хватало
    коротких форм согласия: кандидат отвечал «ага» — и анкета не начиналась.

    Вопросительный знак отменяет согласие: «хочу узнать, какие есть
    варианты?» — это вопрос, а не просьба оформляться."""
    text = user_message.strip().lower()
    if not text:
        return False
    if text.endswith("?"):
        return False
    # Корни слов, по которым согласие видно даже с опечаткой. В боевом
    # тесте кандидат написал «запонляем анкетиу» — по целым словам это не
    # ловится, а по корню «анкет» ловится. Ошибиться тут почти нельзя:
    # вопрос про анкету обычно оканчивается знаком вопроса, а он отсекается
    # выше.
    if any(root in text for root in ("анкет", "заявк", "оформля", "оформи")):
        return True

    words = set(re.findall(r"[\w-]+", text))
    return any(w in words for w in _ANKETA_INTENT)


def _format_kb_context(kb_chunks: list | None) -> str:
    """Найденные факты в текст для промпта, с ограничением объёма.

    Обрезка по KB_CONTEXT_CHAR_LIMIT нужна для предсказуемости расходов:
    без неё удачно сформулированный вопрос, зацепивший пять длинных чанков
    (перечень регионов, список противопоказаний), раздувал бы промпт в разы
    против обычного. Чанки отсортированы поиском по убыванию близости,
    поэтому обрезается всегда наименее релевантный хвост.

    В промпт идут только заголовок и текст. Ключевые слова, по которым чанк
    нашёлся, остаются в индексе — модели они не нужны и только тратили бы
    токены (см. backend/kb_chunker.py)."""
    if not kb_chunks:
        return ""

    lines = []
    used = 0
    for chunk in kb_chunks:
        line = f"- {chunk['question']}: {chunk['answer']}"
        if used + len(line) > KB_CONTEXT_CHAR_LIMIT and lines:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def _build_system_prompt(agent_prompt: str, kb_chunks: list | None = None) -> str:
    """Собирает системный промпт под конкретный вопрос кандидата.

    Промпт состоит из двух частей, и разделение между ними принципиальное:

    1. agent_prompt (data/system_prompt.md) — ТОЛЬКО правила поведения: роль,
       тон, длина ответа, запреты, маркер начала анкеты. Никаких фактов о
       компании. Эта часть неизменна и уходит в каждый запрос.
    2. Факты, найденные ПО ВОПРОСУ кандидата в базе знаний — несколько
       коротких фрагментов из data/kb/, отобранных отдельным векторным
       поиском (см. get_answer и backend/kb_chunker.py).

    Раньше факты жили прямо в system_prompt.md: полный перечень условий,
    зарплат, нормативов и адресов уходил в GigaChat при каждом сообщении,
    включая «здравствуйте». Сейчас неизменная часть содержит только правила,
    а конкретика приезжает адресно — под заданный вопрос.

    Если релевантных фактов не нашлось, секция просто не появляется. Модель
    в этом случае обязана сказать, что информации нет, и предложить уточнить
    у менеджера — это прописано в system_prompt.md."""
    context = _format_kb_context(kb_chunks)
    if not context:
        return agent_prompt

    return (
        f"{agent_prompt}\n\n"
        f"## Факты из базы знаний по вопросу кандидата\n"
        f"{context}\n\n"
        f"Отвечай только на основании этих фактов. Не добавляй цифры, сроки и "
        f"условия, которых здесь нет. Если для ответа данных не хватает — скажи "
        f"об этом прямо и предложи уточнить у менеджера."
    )


def _reload_all_locked():
    with open(AGENT_PROMPT_PATH, "r", encoding="utf-8") as f:
        agent_prompt = f.read().strip()

    new_index, new_metadata = load_index(INDEX_PATH, META_PATH)

    # KB-индекс, в отличие от FAQ-индекса выше, оборачиваем в try/except:
    # load_index() бросает RuntimeError, если файла нет на диске (см.
    # rag_index.py) — для FAQ-индекса это оправданно (без него бот
    # физически не может отвечать, пусть падает явно), но KB-индекс —
    # более новая функция (собирается build_index.py при следующей
    # пересборке, см. run_all.py/start.sh); на самой первой сборке проекта
    # ПОСЛЕ этой правки, до первого запуска build_index.py, файла ещё не
    # будет. Явный RuntimeError здесь уронил бы весь сервер только из-за
    # отсутствия опциональной оптимизации — вместо этого просто отключаем
    # KB-поиск (get_answer работает без него, как и раньше до этой правки).
    try:
        new_kb_index, new_kb_metadata = load_index(KB_INDEX_PATH, KB_META_PATH)
    except RuntimeError:
        new_kb_index, new_kb_metadata = None, None
        print(
            "[assistant] KB-индекс не найден (ожидается после следующего "
            "запуска `python -m backend.build_index`) — поиск по базе "
            "знаний временно отключён, отвечает только FAQ-поиск/GigaChat."
        )

    # Кешируем СЫРОЙ agent_prompt, а не готовую строку
    # системного промпта — сам промпт теперь собирается заново под каждого
    # конкретного кандидата в get_answer() (см. _build_system_prompt), в
    # зависимости от того, какая вакансия ему уже известна/интересна. Чтение
    # с диска и парсинг JSON — единственная тяжёлая часть, которую есть
    # смысл кешировать между запросами; сама сборка строки промпта из уже
    # распарсенного словаря — дёшево, её можно делать каждый раз.
    _state["agent_prompt"] = agent_prompt
    _state["index"] = new_index
    _state["metadata"] = new_metadata
    _state["kb_index"] = new_kb_index
    _state["kb_metadata"] = new_kb_metadata
    _state["mtimes"] = _get_mtimes()

    print("[assistant] База знаний и промпт (пере)загружены.")


def _ensure_fresh():
    with _state_lock:
        current_mtimes = _get_mtimes()
        if current_mtimes != _state["mtimes"]:
            _reload_all_locked()


with _state_lock:
    _reload_all_locked()


def embed_text(text: str) -> np.ndarray:
    """Вектор вопроса кандидата для поиска в FAISS.

    normalize_embeddings=True обязателен и ДОЛЖЕН совпадать с тем, как
    строился индекс (build_index.py). У sentence-transformers значение по
    умолчанию — False, и раньше нормализации не было ни здесь, ни при сборке
    индекса: длины векторов этой модели заметно больше единицы, поэтому
    квадрат расстояния между двумя даже близкими по смыслу вопросами
    оказывался в разы выше порога 1.0, search_similar почти всегда возвращал
    пустой список, и вся экономия на search-first не работала — каждое
    сообщение уходило в GigaChat. После нормализации длина вектора равна 1, и
    порог получает понятный смысл через косинусную близость (см. комментарий
    у FAQ_MIN_SIMILARITY выше).

    ВАЖНО: менять этот флаг в одиночку нельзя — индекс и запрос должны
    строиться одинаково. При изменении поднимите EMBEDDING_VERSION в
    build_index.py, чтобы индекс гарантированно пересобрался."""
    vector = get_embedding_model().encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vector.astype("float32")


def _encrypt_content(content: str) -> str:
    """Шифрует только текст сообщения ("content"), не всю запись целиком —
    "role" ("user"/"assistant") не персональные данные, оставлен открытым,
    чтобы файл истории оставался человекочитаемым по структуре диалога при
    беглом просмотре (кто говорил, сколько реплик), даже если сам текст
    реплик зашифрован."""
    if not content:
        return content
    return crypto_utils.encrypt_bytes(content.encode("utf-8")).decode("ascii")


def _decrypt_content(content: str) -> str:
    if not content:
        return content
    try:
        content.encode("ascii")
    except UnicodeEncodeError:
        # Не-ASCII символы — физически не Fernet-токен, значит открытый
        # текст с прошлого формата (до включения шифрования истории).
        return content
    try:
        return crypto_utils.decrypt_bytes(content.encode("ascii")).decode("utf-8")
    except crypto_utils.EncryptionKeyInvalid:
        # Другой ключ либо повреждённые данные — не роняем всю историю
        # диалога из-за одного нечитаемого сообщения.
        return f"[не удалось расшифровать: {content}]"


def _load_history(source: str, external_id: str) -> list:
    path = paths.conversation_path(source, external_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    for message in history:
        if "content" in message:
            message["content"] = _decrypt_content(message["content"])
    return history


def _save_history(source: str, external_id: str, history: list):
    path = paths.conversation_path(source, external_id)
    encrypted_history = [
        {**message, "content": _encrypt_content(message.get("content", ""))}
        for message in history
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(encrypted_history, f, ensure_ascii=False, indent=2)


# --- Поиск служебных маркеров ---
#
# ВАЖНО: модель НЕ гарантирует, что маркер стоит на отдельной строке — часто
# лепит его в конец обычной фразы без переноса строки, например:
#   "Теперь, как ваше имя? ###SAVE_FIELD:Имя"
# Поэтому маркер ищется ГДЕ УГОДНО в тексте (не только в начале строки).
#
# Два паттерна, в порядке применения:
#
# 1) CLOSED — маркер с явным закрывающим ###. Значение ограничено [^\n#] —
#    не выходит за пределы строки и останавливается на первом же ###, поэтому
#    НЕ поглощает текст, который идёт после маркера в том же сообщении.
#    Без этого ограничения (просто .+? до конца строки/текста) наблюдался
#    реальный баг: "###SAVE_FIELD:Имя=Иван### Теперь, скажите..." — модель
#    закрыла маркер сразу после значения, но старая регулярка всё равно
#    дотягивалась до конца сообщения и записывала в поле "Иван### Теперь,
#    скажите, пожалуйста, вашу фамилию?" — а пользователю уходило пустое
#    сообщение (см. candidates.csv кандидата 2, поле "Имя").
#
# 2) OPEN — запасной вариант для маркера БЕЗ закрывающих ### (модель иногда
#    просто не дописывает закрытие в конце сообщения). Съедает до конца
#    СТРОКИ (\n или конец текста), но не дальше — не может поглотить
#    следующий абзац, если он есть.
#
# Оба риска (маркер не на отдельной строке, маркер без закрытия) актуальны
# и для единственного маркера, оставшегося после переделки анкеты ниже —
# паттерны и защита от них не переписывались, только сузился набор
# распознаваемых маркеров с четырёх до одного.
#
# Единственный маркер, оставшийся у GigaChat после переделки анкеты на
# детерминированную стейт-машину (см. backend/anketa.py и обсуждение пункта
# 2 в истории этого проекта: "зачем отправлять паспорт/СНИЛС/ИНН в
# GigaChat, если программа и так видит, заполнено поле или нет").
#
# START_ANKETA — чисто управляющий сигнал ("кандидат согласился начать
# анкету"), НЕ содержит никаких персональных данных вообще, в отличие от
# прежнего SAVE_FIELD:поле=значение, где как раз в промпт уходило само
# значение поля. GigaChat по-прежнему ведёт обычный диалог (FAQ,
# консультации по вакансиям, вопросы про условия) — это не убрано и не
# урезано — просто в какой-то момент естественно предлагает перейти к
# анкете и, когда кандидат соглашается, ставит этот маркер. Дальше —
# согласие 152-ФЗ, все 29 полей, подтверждение карточки, удаление —
# полностью на стороне _handle_anketa_turn/anketa.py, GigaChat к этому не
# возвращается, пока анкета не будет закрыта (см. get_answer).
CLOSED_MARKER_PATTERN = re.compile(r"#{2,}\s*(START_ANKETA)\s*#{2,}", re.IGNORECASE)
# Без "(?=\n|$)" на конце: раньше маркер вырезался, только если после него
# в строке ничего не было. Стоило модели написать "###START_ANKETA вот
# вопросы" — и маркер оставался в тексте, кандидат видел его глазами, а
# анкета не начиналась.
OPEN_MARKER_PATTERN = re.compile(r"#{2,}\s*(START_ANKETA)\s*#{0,}", re.IGNORECASE)

# Модель иногда (по привычке из markdown) оборачивает маркер в тройные кавычки
# ```...```, хотя промпт этого не просит. Сам маркер вырезается по паттернам
# выше, а после этого от него может остаться пустая пара ```\n```. Эта
# регулярка вырезает ТОЛЬКО пустые/пробельные fence-пары — легитимные code
# block с реальным содержимым внутри не трогает.
EMPTY_FENCE_PATTERN = re.compile(r"```\w*[ \t]*\n?\s*```")


def _process_markers(raw_text: str) -> tuple[str, bool]:
    """
    Находит маркер START_ANKETA в тексте ответа GigaChat (в любом месте, не
    только на отдельной строке), убирает его из текста, который увидит
    кандидат.

    Возвращает (текст_без_маркера, анкета_запрошена):
    - анкета_запрошена — True, если в этом ответе появился START_ANKETA —
      используется в get_answer, чтобы на СЛЕДУЮЩЕМ ходу сразу передать
      диалог в анкетную ветку (_handle_anketa_turn), а не снова в GigaChat.
    """
    anketa_requested = False

    def handle_marker(marker_body: str) -> str:
        nonlocal anketa_requested
        if marker_body.strip().upper() == "START_ANKETA":
            anketa_requested = True
        return ""

    text = CLOSED_MARKER_PATTERN.sub(lambda m: handle_marker(m.group(1)), raw_text)
    text = OPEN_MARKER_PATTERN.sub(lambda m: handle_marker(m.group(1)), text)
    text = EMPTY_FENCE_PATTERN.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # схлопываем пустые строки, оставшиеся от вырезанного маркера

    clean_text = text.strip()

    if not clean_text:
        # Модель прислала ответ, состоящий ТОЛЬКО из маркера, без единого
        # слова для кандидата. Это баг промпта (модель обязана сопровождать
        # маркер текстом), не маскируем его красивой заглушкой, а громко
        # логируем, чтобы это было видно в консоли и не потерялось молча.
        print(
            f"[assistant] ВНИМАНИЕ: ответ модели стал пустым после зачистки "
            f"маркера. Сырой ответ модели: {raw_text!r}"
        )

    return clean_text, anketa_requested


def _handle_anketa_turn(source: str, external_id: str,
                        user_message: str, last_bot_message: str,
                        anketa_start_requested: bool = False):
    """Обрабатывает один ход диалога в рамках анкеты.

    Шаг диалога берётся из карточки кандидата (candidates.get_stage), а НЕ
    угадывается по тексту предыдущего сообщения бота. Раньше переходы были
    завязаны на точные фразы вроде startswith("Проверьте, пожалуйста, все
    данные"): любая косметическая правка текста молча ломала логику, а
    вдобавок делала невозможным исправление поля — после подсказки «назовите
    поле и значение» фраза уже не совпадала, и бот показывал карточку по
    кругу, бесконечно.

    last_bot_message оставлен в сигнатуре только ради одного случая: самый
    первый показ текста согласия 152-ФЗ, когда карточки ещё нет и хранить
    шаг физически негде.

    Возвращает текст ответа, если ход обработан анкетной веткой (GigaChat в
    этом случае не вызывается вообще), либо None — значит идёт обычный
    диалог.
    """
    candidate_id = candidates.find_candidate(source, external_id)
    stage = candidates.get_stage(candidate_id) if candidate_id else anketa.STAGE_NONE

    # Карточка с прежним набором полей: продолжать её нельзя, вопросы и
    # сохранённые ответы разъехались. Предлагаем начать заново, вместо того
    # чтобы молча сбиться на середине. Удаление при этом остаётся доступным.
    # STAGE_DELETE исключён наравне с остальными: если человек уже попросил
    # удалить данные и ему показали предупреждение, подтверждение «да, удали»
    # должно сработать. Иначе удаление обрывалось на полпути сообщением про
    # обновлённую форму, и данные оставались на диске.
    if (candidate_id
            and stage not in (anketa.STAGE_NONE, anketa.STAGE_DONE, anketa.STAGE_DELETE)
            and candidates.is_outdated(candidate_id)
            and not anketa.is_delete_request(user_message)):
        candidates.set_stage(candidate_id, anketa.STAGE_DONE)
        return (
            "Извините, форма анкеты обновилась, и прежние ответы уже не "
            "подходят. Давайте заполним заново — напишите «хочу оставить "
            "заявку», это займёт несколько минут."
        )

    # --- Карточки ещё нет: согласие 152-ФЗ ---
    if not candidate_id:
        if (last_bot_message.endswith(anketa.LAW_CONSENT_TEXT)
                or is_awaiting_consent(source, external_id)):
            return _handle_law_answer(source, external_id, user_message)
        # Прямая просьба начать анкету — распознаётся кодом, без нейросети.
        # Ноль токенов, мгновенно и без зависимости от того, поставит ли
        # модель нужный маркер (в тесте она его ставила через раз).
        if anketa_start_requested or anketa.is_anketa_request(user_message):
            _mark_awaiting_consent(source, external_id)
            return anketa.LAW_CONSENT_TEXT
        return None

    # --- Подтверждение удаления данных ---
    if stage == anketa.STAGE_DELETE:
        if anketa.is_delete_confirmed(user_message):
            candidates.delete_candidate(candidate_id)
            return anketa.DELETE_DONE_MESSAGE

        # Человек повторил просьбу вместо подтверждения: «удали мои данные»
        # ещё раз. Люди так делают постоянно — перефразируют, а не читают
        # инструкцию дословно. Раньше это считалось отказом от удаления, и
        # кандидата возвращали в анкету, хотя он второй раз попросил стереть
        # данные. Переспрашиваем, но с шага не уходим.
        if anketa.is_delete_request(user_message):
            return anketa.DELETE_CONFIRMATION_PROMPT
        # Передумал удалять — возвращаем на тот шаг, где прервались.
        if candidates.is_card_complete(candidate_id):
            candidates.set_stage(candidate_id, anketa.STAGE_CONFIRM)
            return anketa.format_card_for_confirmation(candidate_id)
        candidates.set_stage(candidate_id, anketa.STAGE_FIELDS)
        next_step = anketa.get_next_step(candidate_id)
        return next_step[1] if next_step else anketa.format_card_for_confirmation(candidate_id)

    # --- Согласие ещё не получено (карточка осталась от прежней версии) ---
    if stage == anketa.STAGE_LAW or not candidates.is_law_acknowledged(candidate_id):
        # Просьба удалить данные работает и здесь. Раньше ветка согласия
        # стояла раньше проверки удаления, и на этом шаге «удали мои данные»
        # получало переспрос про 152-ФЗ вместо удаления.
        if anketa.is_delete_request(user_message):
            candidates.set_stage(candidate_id, anketa.STAGE_DELETE)
            return anketa.DELETE_CONFIRMATION_PROMPT
        if last_bot_message.endswith(anketa.LAW_CONSENT_TEXT) or stage == anketa.STAGE_LAW:
            return _handle_law_answer(source, external_id, user_message)
        if anketa_start_requested or anketa.is_anketa_request(user_message):
            candidates.set_stage(candidate_id, anketa.STAGE_LAW)
            return anketa.LAW_CONSENT_TEXT
        return None

    # Команда удаления доступна ВСЕГДА, пока карточка существует, — в том
    # числе после того, как анкета уже отправлена HR.
    #
    # Проверка стоит ВЫШЕ выхода в обычный диалог намеренно. Раньше она была
    # ниже, и после подтверждения анкеты просьба «удали мои данные»
    # проваливалась в GigaChat: та отвечала что-то общее, а данные
    # оставались на месте. Именно этот момент важнее всего: право отозвать
    # согласие по 152-ФЗ возникает как раз ПОСЛЕ передачи данных, а не до
    # неё, и README прямо обещает кандидату такую возможность.
    if anketa.is_delete_request(user_message):
        candidates.set_stage(candidate_id, anketa.STAGE_DELETE)
        return anketa.DELETE_CONFIRMATION_PROMPT

    # --- Анкета уже принята ---
    if stage == anketa.STAGE_DONE or _card_already_confirmed(candidate_id):
        # «Я заполнил?», «что осталось» — отвечает КОД, а не нейросеть.
        if anketa.is_anketa_status_question(user_message):
            return anketa.format_status(candidate_id, anketa.STAGE_DONE)

        # Заново — ТОЛЬКО по явной просьбе человека: кнопка или прямая
        # фраза. Метку от нейросети здесь НЕ слушаем намеренно.
        #
        # Раньше слушали, и выходил круг: человек спрашивал «я заполнил?»,
        # модель выдумывала «осталось заполнить здоровье», человек отвечал
        # «да», модель ставила метку, и код показывал согласие 152-ФЗ
        # заново — хотя анкета давно сдана. Дальше человек писал что-то со
        # словом «нет», и это читалось как отказ от согласия.
        if anketa.is_anketa_request(user_message):
            candidates.set_stage(candidate_id, anketa.STAGE_LAW)
            return anketa.LAW_CONSENT_TEXT
        return None

    # --- Показана карточка, ждём подтверждения или правки ---
    if stage == anketa.STAGE_CONFIRM:
        # Разбор правки идёт ПЕРЕД проверкой согласия. «Да, но телефон
        # 8999...» — это просьба исправить, а не подтверждение, хотя слово
        # «да» в ней есть. Раньше такая фраза закрывала анкету со старым
        # телефоном, и HR не мог дозвониться.
        correction = anketa.parse_correction(user_message)
        if correction:
            field, value = correction
            try:
                candidates.set_field(candidate_id, field, value)
            except FieldValidationError as e:
                return f"{e.message}\n\nНапишите это поле ещё раз, например: «{field} ...»"
            return (
                f"Исправил: {field} — {value}.\n\n"
                + anketa.format_card_for_confirmation(candidate_id)
            )

        if anketa.is_confirmation_yes(user_message):
            candidates.mark_card_confirmed(candidate_id)
            candidates.set_stage(candidate_id, anketa.STAGE_DONE)
            fresh_card = candidates.get_card(candidate_id)
            notifications.send_hr_notification(
                notifications.format_new_candidate_notification(fresh_card),
                buttons=[{
                    "text": "Показать все данные кандидата",
                    "callback_data": f"card:{candidate_id}",
                }],
            )
            return (
                "Спасибо, анкета принята! С вами свяжется менеджер — обсудит "
                "детали и купит билет до места обучения. Если появятся "
                "вопросы, пишите, я на связи."
            )

        return (
            "Если нужно что-то исправить — напишите поле и новое значение. "
            "Можно скопировать строку из анкеты выше и вставить с "
            "исправленным значением, например: «Телефон 89991234567».\n\n"
            "Если всё верно — напишите «да»."
        )

    # --- Спросили, приостановить ли анкету ---
    if stage == anketa.STAGE_PAUSE_ASK:
        if anketa.is_law_accepted(user_message):
            candidates.set_stage(candidate_id, anketa.STAGE_PAUSED)
            return (
                "Хорошо, анкету отложил. Спрашивайте что угодно о работе.\n\n"
                + anketa.PAUSED_REMINDER
            )
        if anketa.is_law_declined(user_message):
            candidates.set_stage(candidate_id, anketa.STAGE_FIELDS)
            step = anketa.get_next_step(candidate_id)
            return step[1] if step else anketa.format_card_for_confirmation(candidate_id)
        # Ответ не разобран — повторяем выбор, а не гадаем.
        return anketa.PAUSE_ASK_TEXT

    # --- Анкета на паузе: обычный диалог, пока не попросят вернуться ---
    if stage == anketa.STAGE_PAUSED:
        if anketa.is_resume_request(user_message):
            candidates.set_stage(candidate_id, anketa.STAGE_FIELDS)
            step = anketa.get_next_step(candidate_id)
            if step is None:
                candidates.set_stage(candidate_id, anketa.STAGE_CONFIRM)
                return anketa.format_card_for_confirmation(candidate_id)
            return f"Возвращаемся к анкете.\n\n{step[1]}"
        # Вопрос уходит в обычный диалог. Напоминание припишется в
        # get_answer, чтобы человек не думал, что анкета потерялась.
        return None

    # --- Основной сбор полей ---
    if anketa.is_progress_question(user_message):
        return anketa.format_progress_answer(candidate_id)

    # Нажали кнопку «Заполнить анкету», уже находясь в анкете. Раньше текст
    # кнопки уходил в поле как ответ: на скриншоте он попал в дату рождения,
    # человек получил «не понял дату» и застрял.
    if anketa.is_anketa_request(user_message):
        return anketa.format_status(candidate_id, anketa.STAGE_FIELDS)

    next_step = anketa.get_next_step(candidate_id)
    if next_step:
        field, _ = next_step
        ok, message = anketa.process_answer(candidate_id, field, user_message)
        if ok and message is None:
            candidates.set_stage(candidate_id, anketa.STAGE_CONFIRM)
            return anketa.format_card_for_confirmation(candidate_id)

        # Ответ не подошёл. Проверяем ПОСЛЕ неудачной проверки, а не до:
        # иначе валидный ответ вроде «Последнее место работы: заполнял
        # анкеты» был бы съеден как вопрос и потерян.
        if not ok and anketa.is_anketa_status_question(user_message):
            return anketa.format_status(candidate_id, anketa.STAGE_FIELDS)

        # Человек задал вопрос вместо ответа на пункт. Раньше бот просто
        # повторял вопрос с текстом ошибки, и выглядело так, будто его
        # вопрос проигнорировали. Теперь предлагаем выбор явно.
        if not ok and anketa.looks_like_question(user_message):
            candidates.set_stage(candidate_id, anketa.STAGE_PAUSE_ASK)
            return anketa.PAUSE_ASK_TEXT

        return message

    candidates.set_stage(candidate_id, anketa.STAGE_CONFIRM)
    return anketa.format_card_for_confirmation(candidate_id)


def _handle_law_answer(source: str, external_id: str, user_message: str):
    """Разбор ответа на текст согласия 152-ФЗ.

    Согласие должно быть ЯВНЫМ: непонятный ответ приводит к переспросу, а не
    к молчаливому согласию по умолчанию. Карточка кандидата создаётся только
    здесь, в момент фактического согласия, — отказавшийся не оставляет после
    себя строки в таблице персональных данных."""
    if anketa.is_law_declined(user_message):
        _clear_awaiting_consent(source, external_id)
        return (
            "Хорошо, анкету отложим. Спрашивайте о работе — расскажу про "
            "вакансии, зарплату, вахту, обучение и документы. Когда "
            "захотите оставить заявку, просто скажите об этом."
        )

    if not anketa.is_law_accepted(user_message):
        # Человек задал вопрос вместо «да» или «нет». Пропускаем его в
        # обычный диалог, а шаг согласия оставляем в силе — к ответу
        # припишется короткое напоминание (см. get_answer).
        #
        # Раньше здесь безусловно возвращался повтор согласия: на вопрос
        # «зарплата?» человек получал юридический текст вместо ответа, и
        # так по кругу — выйти было нельзя.
        if anketa.looks_like_question(user_message):
            return None
        return anketa.LAW_CONSENT_RETRY

    _clear_awaiting_consent(source, external_id)
    candidate_id = candidates.get_or_create_candidate(source, external_id)
    candidates.mark_law_acknowledged(candidate_id)
    candidates.set_stage(candidate_id, anketa.STAGE_FIELDS)

    first_step = anketa.get_next_step(candidate_id)
    if first_step is None:
        candidates.set_stage(candidate_id, anketa.STAGE_CONFIRM)
        return anketa.format_card_for_confirmation(candidate_id)
    return first_step[1]


def _card_already_confirmed(candidate_id: str) -> bool:
    card = candidates.get_card(candidate_id)
    return bool(card.get("Факт ознакомления с готовой карточкой", "").strip())


# Служебная подсказка о заполненных полях БОЛЬШЕ НЕ ПЕРЕДАЁТСЯ модели.
#
# Раньше в каждый запрос уходила строка «кандидат сообщил 27 пункт(ов)
# анкеты» со списком полей и просьбой не упоминать их вслух. Модель
# упоминала — и при этом врала: видела 27 из 29 и сочиняла «осталось
# добавить информацию о здоровье», хотя всё было заполнено. Человек верил
# и лез доделывать несуществующее, а дальше диалог уходил в круг.
#
# Принцип, к которому пришли: НЕЙРОСЕТЬ НИЧЕГО НЕ ЗНАЕТ ПРО АНКЕТУ И
# НИКОГДА ПРО НЕЁ НЕ ГОВОРИТ. Всё, что касается формы — статус, что
# осталось, заполнена или нет, начать заново — отвечает код
# (anketa.format_status). Врать становится нечем.
#
# Побочный выигрыш: подсказка уходила в КАЖДЫЙ запрос и стоила токенов,
# а вопросы про анкету — самые частые после сдачи — теперь до модели
# вообще не доходят.


# Отдельная блокировка на КАЖДЫЙ диалог. _history_lock защищает только сами
# операции чтения и записи файла, но между чтением истории в начале
# get_answer и записью в конце проходит вызов GigaChat — несколько секунд.
# Два сообщения одного человека, отправленные в этот промежуток (две
# вкладки сайта, быстрый повтор в Telegram), читали одну и ту же историю и
# записывали её по очереди: сообщение того, кто закончил первым, пропадало.
#
# Общая блокировка на всех тут не годится — она выстроила бы в очередь
# вообще всех кандидатов на время каждого обращения к модели. Блокировка
# на диалог сериализует только сообщения одного человека, а это ровно то,
# что и должно быть: два своих сообщения он всё равно ждёт по очереди.
# Кто сейчас стоит на шаге согласия по 152-ФЗ.
#
# До согласия карточки кандидата НЕТ — мы её намеренно не создаём, пока
# человек не согласился. Значит и этап хранить негде: в карточке нельзя,
# а по тексту предыдущего сообщения бота — ненадёжно, потому что между
# показом согласия и ответом человек может задать вопрос, и последним
# сообщением бота станет ответ на него.
#
# Здесь только пара «канал + номер диалога», персональных данных нет.
# Потеря при перезапуске безобидна: человек просто нажмёт кнопку снова.
_awaiting_consent: "OrderedDict[str, float]" = OrderedDict()
_MAX_AWAITING = 2000


def _mark_awaiting_consent(source: str, external_id: str) -> None:
    key = f"{source}:{external_id}"
    with _session_locks_guard:
        if len(_awaiting_consent) >= _MAX_AWAITING:
            for _ in range(_MAX_AWAITING // 4):
                _awaiting_consent.popitem(last=False)
        _awaiting_consent[key] = time.time()


def is_awaiting_consent(source: str, external_id: str) -> bool:
    return f"{source}:{external_id}" in _awaiting_consent


def _clear_awaiting_consent(source: str, external_id: str) -> None:
    with _session_locks_guard:
        _awaiting_consent.pop(f"{source}:{external_id}", None)


_session_locks: "OrderedDict[str, threading.Lock]" = OrderedDict()
_session_locks_guard = threading.Lock()


# Сколько замков держим в памяти. Замок — это несколько десятков байт, но
# диалогов за годы работы накапливаются десятки тысяч, и словарь рос бы без
# конца. Старые записи вычищаются: если диалог давно закончился, замок ему
# больше не нужен, а если кандидат вернётся — создастся заново.
_MAX_SESSION_LOCKS = 2000


def _session_lock(source: str, external_id: str) -> threading.Lock:
    """Замок на конкретный диалог.

    Вытесняются замки диалогов, которыми ДОЛЬШЕ ВСЕГО не пользовались, а не
    просто самые старые по времени создания.

    Разница существенная. Кандидат может заполнять анкету час — его замок
    создан давно, но используется прямо сейчас. При вытеснении по порядку
    создания такой замок вылетал бы первым, на его месте создавался новый
    объект, и два сообщения одного человека могли обработаться параллельно
    — с гонкой за файл истории. Порядок реплик в переписке мог перепутаться.

    move_to_end переносит ключ в конец при каждом обращении, поэтому в
    начале словаря всегда лежит то, что давно не трогали."""
    key = f"{source}:{external_id}"
    with _session_locks_guard:
        existing = _session_locks.get(key)
        if existing is not None:
            _session_locks.move_to_end(key)
            return existing

        if len(_session_locks) >= _MAX_SESSION_LOCKS:
            for _ in range(_MAX_SESSION_LOCKS // 4):
                _session_locks.popitem(last=False)

        lock = threading.Lock()
        _session_locks[key] = lock
        return lock


def get_answer(user_message: str, source: str, external_id: str, top_k: int = 3):
    """
    source — "site" или "telegram".
    external_id — стабильный идентификатор диалога (session_id сайта или chat_id Telegram).

    Обёртка над _get_answer: держит блокировку диалога на всё время
    обработки, чтобы два одновременных сообщения одного кандидата не
    затирали историю друг друга (см. комментарий к _session_locks выше).
    """
    with _session_lock(source, external_id):
        return _get_answer(user_message, source, external_id, top_k)


def _get_answer(user_message: str, source: str, external_id: str, top_k: int = 3):
    start_time = time.time()
    _ensure_fresh()

    # find_candidate, а НЕ get_or_create_candidate: карточка заводится только
    # в момент согласия по 152-ФЗ (см. _handle_anketa_turn). Раньше строка в
    # candidates.csv создавалась на КАЖДОГО, кто написал боту хоть слово —
    # человек спрашивал "какие есть вакансии" и уходил, а в таблице оставалась
    # пустая карточка и запись в candidate_sessions.json.
    candidate_id = candidates.find_candidate(source, external_id)

    with _history_lock:
        history = _load_history(source, external_id)

    # --- Анкетная ветка: полностью без GigaChat --------------------------
    #
    # Раньше вся анкета (включая распознавание согласия 152-ФЗ, самих
    # значений полей и команды удаления) шла через диалог с GigaChat —
    # модель ВИДЕЛА текст кандидата, включая паспорт/СНИЛС/ИНН, чтобы
    # понять, что сохранить (см. историю правок этого файла и обсуждение
    # пункта 2 в этом чате). Ветка ниже — детерминированная стейт-машина
    # (anketa.py): вопрос -> валидатор без ИИ -> следующий вопрос либо
    # дружелюбная ошибка. GigaChat здесь не вызывается ни разу.
    #
    # last_bot_message используется только для одного: чтобы понять, было
    # ли ПРЕДЫДУЩИМ сообщением бота предупреждение об удалении (см.
    # _handle_anketa_turn ниже) — это НЕ хранится как отдельное состояние в
    # candidates.csv (лишнее поле ради однооборотного состояния), достаточно
    # посмотреть последний ответ бота в уже загруженной истории диалога.
    last_bot_message = history[-1]["content"] if history and history[-1].get("role") == "assistant" else ""

    anketa_reply = _handle_anketa_turn(source, external_id, user_message, last_bot_message)
    if anketa_reply is not None:
        response_time_ms = int((time.time() - start_time) * 1000)

        # Если ход оказался удалением данных — историю НЕ сохраняем.
        #
        # Иначе удаление получалось фиктивным: delete_candidate стирает файл
        # переписки с диска, но в памяти этой функции уже лежит загруженная
        # копия последних сообщений — с паспортом, СНИЛС и адресом. Строки
        # ниже записали бы её обратно, и файл возрождался бы через
        # миллисекунду после удаления. То есть ровно те данные, ради которых
        # человек написал «удали», оставались бы на диске.
        if anketa_reply == anketa.DELETE_DONE_MESSAGE:
            history = []
        else:
            history.append({"role": MessagesRole.USER, "content": user_message})
            history.append({"role": MessagesRole.ASSISTANT, "content": anketa_reply})
        # Обрезка истории, как и в остальных ветках ниже. Без неё файл
        # истории рос без ограничения все 29 шагов анкеты и дальше.
        history = history[-MAX_HISTORY_MESSAGES:]
        if history:
            with _history_lock:
                _save_history(source, external_id, history)
        logger.log_interaction(source, external_id, user_message, anketa_reply, response_time_ms, "ok")
        return anketa_reply, []


    with _state_lock:
        current_index = _state["index"]
        current_metadata = _state["metadata"]
        current_kb_index = _state["kb_index"]
        current_kb_metadata = _state["kb_metadata"]
        current_agent_prompt = _state["agent_prompt"]

    query_vec = embed_text(user_message)
    similar_items = search_similar(
        current_index, current_metadata, query_vec, k=top_k, min_similarity=FAQ_MIN_SIMILARITY
    )

    # --- Search-first: FAQ раньше GigaChat -------------------------------
    #
    # Раньше search_similar() использовалась только как ИСТОЧНИК КОНТЕКСТА
    # для GigaChat (её результат подмешивался в промпт, но сам ответ всегда
    # формировала модель, даже если нашёлся точный совпадающий вопрос из
    # faqs.json). Теперь: нашёлся релевантный вопрос (similar_items
    # непустой — search_similar уже фильтрует по порогу схожести
    # max_distance, см. rag_index.py) — отдаём готовый ответ ИЗ FAQ
    # НАПРЯМУЮ, без обращения к GigaChat вообще (промпт даже не строится —
    # см. ниже, построение промпта происходит ПОСЛЕ этой проверки, а не до,
    # чтобы не тратить время на KB-поиск и сборку промпта впустую, если
    # готовый ответ уже найден). Экономит реальные деньги (не тратится
    # вызов GigaChat на вопрос, ответ на который уже есть готовым текстом)
    # и время ответа (без сетевого запроса к GigaChat).
    #
    # Берём ПЕРВЫЙ (самый близкий) результат — top_k=3 по умолчанию
    # существует для контекста, который раньше подмешивался в промпт
    # GigaChat, а не для того, чтобы выбирать между несколькими готовыми
    # ответами: если первый результат прошёл порог схожести, он и есть
    # лучшее совпадение с вопросом кандидата.
    if similar_items:
        faq_answer = similar_items[0]["answer"]
        response_time_ms = int((time.time() - start_time) * 1000)
        history.append({"role": MessagesRole.USER, "content": user_message})
        history.append({"role": MessagesRole.ASSISTANT, "content": faq_answer})
        history = history[-MAX_HISTORY_MESSAGES:]
        with _history_lock:
            _save_history(source, external_id, history)
        logger.log_interaction(source, external_id, user_message, faq_answer, response_time_ms, "ok", "faq_direct")
        return faq_answer, similar_items

    # Не нашли релевантный вопрос в FAQ — обращаемся к GigaChat. Прежде чем
    # строить промпт, делаем ОТДЕЛЬНЫЙ поиск по базе знаний из data/kb/ (см.
    # backend/kb_chunker.py, backend/build_index.py) — раздельный от FAQ-
    # поиска выше индекс, чтобы найденные факты из базы знаний не
    # конкурировали за место в top_k с готовыми FAQ-ответами (которые сюда
    # в принципе не должны попадать вместе с KB-фактами — разное
    # назначение, см. комментарий у KB_INDEX_PATH). Если current_kb_index
    # ещё не собран (None — см. _reload_all_locked) — просто идём с пустым
    # списком чанков, промпт соберётся без раздела базы знаний, как было
    # изначально, до первой сборки KB-индекса.
    kb_chunks = []
    if current_kb_index is not None:
        kb_chunks = search_similar(
            current_kb_index, current_kb_metadata, query_vec, k=KB_TOP_K, min_similarity=KB_MIN_SIMILARITY
        )

    # ВАЖНО: явно добавляем user_message отдельным элементом уже внутри
    # messages ниже — если кандидат называет вакансию ПРЯМО В ЭТОМ
    # сообщении (например, самое первое сообщение диалога), в history его
    # ещё нет; для самого промпта это не нужно (kb_chunks уже найдены по
    # query_vec — векторному представлению именно user_message).
    current_system_prompt = _build_system_prompt(current_agent_prompt, kb_chunks)

    messages = [Messages(role=MessagesRole.SYSTEM, content=current_system_prompt)]

    for turn in history:
        messages.append(Messages(role=turn["role"], content=turn["content"]))

    messages.append(
        Messages(
            role=MessagesRole.USER,
            content=(
                f"Вопрос пользователя: {user_message}"
            ),
        )
    )

    try:
        # X-Session-ID включает кеширование на стороне GigaChat: если контекст
        # (системный промпт + предыдущие сообщения) частично совпадает с
        # прошлым запросом под тем же идентификатором, GigaChat не пересчитывает
        # эту часть заново — платится и считается только то, что реально
        # изменилось (см. https://developers.sber.ru/docs/ru/gigachat/guides/keeping-context,
        # раздел "Кэширование запросов"; ответ API возвращает точное число
        # закешированных токенов в поле precached_prompt_tokens).
        #
        # Используем candidate_id, а не session_id/chat_id: candidate_id
        # одинаково устроен для обоих каналов (веб и Telegram) и стабилен на
        # весь диалог одного кандидата — ровно то, что нужно кешу, чтобы
        # находить совпадение между последовательными сообщениями одного и
        # того же разговора.
        #
        # Экономия не гарантирована на 100% каждый запрос — зависит от того,
        # сколько контекста реально совпало с предыдущим (system-часть внутри
        # одного диалога не меняется между сообщениями кандидата, значит по
        # документации должна кешироваться), и от TTL кеша на стороне
        # GigaChat, который в документации явно не указан числом.
        # candidate_id может ещё не существовать (карточка заводится только
        # при согласии по 152-ФЗ) — тогда берём стабильную пару
        # источник+идентификатор диалога: для кеша важно лишь то, чтобы
        # значение не менялось между сообщениями одного разговора.
        gigachat.context.session_id_cvar.set(candidate_id or f"{source}_{external_id}")

        # verify_ssl_certs=True (по умолчанию у самой библиотеки gigachat)
        # — раньше здесь стояло False, потому что без корневого сертификата
        # НУЦ Минцифры проверка TLS для GigaChat падает с ошибкой
        # "self-signed certificate in certificate chain" (см. setup.py,
        # функция _install_gigachat_certificate — там же подробное
        # объяснение, откуда эта проблема, и почему verify_ssl_certs=False
        # было небезопасным способом её обойти). Теперь сертификат
        # устанавливается один раз при установке проекта (python setup.py),
        # и обычная проверка сертификата работает корректно — если
        # ПОЯВИТСЯ ошибка сертификата снова после этой правки, вероятная
        # причина — setup.py не смог скачать сертификат автоматически
        # (например, недоступен gu-st.ru) и нужно установить его вручную,
        # см. https://developers.sber.ru/docs/ru/gigachat/certificates
        # timeout ОБЯЗАТЕЛЕН. По умолчанию библиотека ждёт ответа сколько
        # угодно, и один зависший вызов держит рабочий процесс сервера
        # занятым. Несколько таких подряд — и бот перестаёт отвечать всем,
        # включая тех, чьи сообщения вообще не требуют нейросети.
        #
        # 30 секунд с запасом: в боевом логе самый долгий честный ответ
        # занял 22,5 секунды. Всё, что дольше, почти наверняка зависание, а
        # не медленный ответ — и лучше отдать кандидату запасной ответ из
        # базы знаний, чем заставлять его смотреть в «Печатает…» минутами.
        with GigaChat(credentials=GIGACHAT_CREDENTIALS, timeout=GIGACHAT_TIMEOUT_SECONDS) as giga:
            response = giga.chat(Chat(messages=messages))
        raw_answer = response.choices[0].message.content
        status = "ok"
        error_comment = ""

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        precached_tokens = getattr(usage, "precached_prompt_tokens", None) if usage else None
        if precached_tokens:
            print(f"[assistant] GigaChat: {precached_tokens} токенов взято из кеша (не тарифицировано).")
    except Exception as e:
        # ВАЖНО: кандидат НИКОГДА не должен увидеть, что что-то сломалось —
        # ни слов "ошибка", ни "API", ни "лимит", ни любого другого признака
        # технической неисправности (решение продукта). При любом сбое
        # GigaChat бот просто отвечает на основе того, что уже нашлось в FAQ
        # (search_similar чуть выше по функции), как обычный ответ по базе,
        # а не как сообщение о проблеме.
        #
        # Для администратора при этом причина различается точно (по типу
        # исключения библиотеки gigachat, а не по угадыванию текста ошибки),
        # чтобы в logs.csv было по-настоящему понятно, что случилось —
        # неверный ключ это, исчерпанный лимит запросов или что-то ещё.
        if isinstance(e, AuthenticationError):
            error_kind = "gigachat_auth"
            print(
                f"[assistant] GigaChat: ошибка авторизации — проверьте "
                f"GIGACHAT_CREDENTIALS в .env (ключ неверный или истёк). {e}"
            )
        elif isinstance(e, RateLimitError):
            error_kind = "gigachat_rate_limit"
            print(f"[assistant] GigaChat: исчерпан лимит запросов к API (429). {e}")
        elif isinstance(e, (ResponseError, GigaChatException)):
            error_kind = "gigachat_api"
            print(f"[assistant] GigaChat: ошибка на стороне API. {e}")
        else:
            error_kind = "gigachat_unknown"
            print(f"[assistant] GigaChat: непредвиденная ошибка соединения. {e}")

        # В журнал — отдельной записью. Обычная строка диалога ниже покажет,
        # что кандидат получил ответ, но не скажет, что нейросеть при этом
        # упала: ответ-то он получит, просто запасной. Без этой записи сбои
        # GigaChat на сервере были бы не видны вообще.
        logger.log_event(
            logger.EVENT_GIGACHAT,
            f"Сбой обращения к GigaChat ({error_kind}): {type(e).__name__}. {e}"[:400],
            status=logger.STATUS_ERROR,
            source=source, external_id=external_id,
        )

        # Откат на базу знаний. similar_items здесь ВСЕГДА пуст — непустой
        # результат FAQ-поиска вернул бы ответ гораздо раньше, до обращения к
        # GigaChat. Раньше проверка стояла именно на similar_items, то есть
        # ветка не срабатывала никогда, и кандидат при любом сбое модели
        # получал безликую заглушку, хотя подходящий факт был найден.
        #
        # А вот kb_chunks к этому моменту непуст, если поиск по базе знаний
        # что-то нашёл, — их и отдаём. Ответ получится суховатым, без
        # пересказа модели, но по существу и с верными цифрами, что заметно
        # лучше «не могу сформулировать». Ровно этот случай и был у вас,
        # когда GigaChat падал на SSL-сертификате.
        if kb_chunks:
            raw_answer = "\n\n".join(chunk["answer"] for chunk in kb_chunks[:2])
        else:
            # В FAQ ничего релевантного не нашлось — честно, но БЕЗ
            # технических слов: как будто бот действительно не знает ответ,
            # а не "сломался".
            raw_answer = (
                "Пока не могу сформулировать точный ответ на этот вопрос. "
                "Уточните, пожалуйста, что именно вас интересует, либо "
                "свяжитесь с менеджером напрямую — контакты есть на сайте."
            )
        status = "error"
        error_comment = f"{error_kind}: {e}"
        prompt_tokens = completion_tokens = total_tokens = precached_tokens = None

    clean_answer, anketa_start_requested = _process_markers(raw_answer)

    if anketa_start_requested:
        # GigaChat только что поставила START_ANKETA — кандидат согласился
        # начать анкету ПРЯМО В ЭТОМ сообщении. Не отдаём кандидату голый
        # ответ GigaChat (что-то вроде "Отлично, давайте начнём!") без
        # продолжения — сразу же, в этом же ходу, показываем текст согласия
        # 152-ФЗ (см. _handle_anketa_turn, anketa_start_requested=True),
        # объединяя его с ответом GigaChat в одно сообщение кандидату.
        if not _user_really_asked_for_anketa(user_message):
            # Модель поставила маркер там, где кандидат ни о чём таком не
            # просил. Игнорируем: пусть ответ останется обычным ответом на
            # вопрос. В лог пишем, чтобы такие случаи было видно и можно
            # было поправить формулировку в system_prompt.md.
            print(
                f"[assistant] START_ANKETA проигнорирован: в сообщении "
                f"кандидата нет согласия ({user_message[:60]!r})."
            )
            # Текст кандидата уходит в ЗАШИФРОВАННУЮ колонку: он написан
            # человеком, а значит это персональные данные, и в открытом
            # комментарии ему не место.
            logger.log_event(
                logger.EVENT_MARKER,
                "Модель предложила анкету, но согласия в сообщении не было — отклонено.",
                status=logger.STATUS_WARN,
                source=source, external_id=external_id,
                details=user_message,
            )
            consent_text = None
        else:
            consent_text = _handle_anketa_turn(
                source, external_id, user_message, last_bot_message,
                anketa_start_requested=True,
            )
        if consent_text:
            clean_answer = f"{clean_answer}\n\n{consent_text}" if clean_answer else consent_text

    # Пустой ответ кандидату уходить не должен ни при каких обстоятельствах.
    #
    # Так бывает, когда модель прислала ОДИН маркер без текста, а маркер мы
    # отклонили (см. защиту от навязанной анкеты выше). Человек видел в чате
    # «Пустой ответ сервера» — то есть явно сломанного бота, хотя сломанного
    # ничего не было.
    if not clean_answer.strip():
        clean_answer = (
            "Расскажите, пожалуйста, что именно вас интересует по работе — "
            "вакансии, зарплата, вахта, обучение или документы."
        )
        logger.log_event(
            logger.EVENT_GIGACHAT,
            "Модель вернула пустой ответ — отправлен запасной текст.",
            status=logger.STATUS_WARN,
            source=source, external_id=external_id,
        )

    # Человек стоит на шаге согласия и задал вопрос. Ответили — и напомнили
    # одной строкой, что анкета ждёт. Полный юридический текст не повторяем:
    # он уже был показан, а второй раз читать его никто не станет.
    if is_awaiting_consent(source, external_id):
        clean_answer = f"{clean_answer}\n\n{anketa.LAW_CONSENT_REMINDER}"
    elif (candidate_id
            and candidates.get_stage(candidate_id) == anketa.STAGE_PAUSED):
        clean_answer = f"{clean_answer}\n\n{anketa.PAUSED_REMINDER}"

    history.append({"role": MessagesRole.USER, "content": user_message})
    history.append({"role": MessagesRole.ASSISTANT, "content": clean_answer})
    history = history[-MAX_HISTORY_MESSAGES:]

    with _history_lock:
        _save_history(source, external_id, history)

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.log_interaction(
        source=source,
        external_id=external_id,
        query=user_message,
        response=clean_answer,
        response_time_ms=elapsed_ms,
        status=status,
        comment=error_comment,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=precached_tokens,
    )

    return clean_answer, similar_items


def get_history(source: str, external_id: str, limit: int = 40) -> list:
    """Переписка для показа на странице при её открытии.

    Нужна потому, что сервер помнил диалог, а страница — нет. Человек
    сворачивал вкладку, открывал заново, видел свежее приветствие и думал,
    что разговор начался с нуля. А на сервере он всё ещё стоял посреди
    анкеты — и его «привет» уходило прямо в поле «Фамилия». В уведомлении
    HR так и появился кандидат с фамилией «привет».

    Возвращает список вида [{"role": "user"|"assistant", "text": ...}] —
    уже расшифрованный и без служебных полей."""
    with _history_lock:
        history = _load_history(source, external_id)

    result = []
    for message in history[-limit:]:
        role = str(message.get("role", ""))
        text = str(message.get("content", "")).strip()
        if not text:
            continue
        result.append({
            "role": "user" if "user" in role.lower() else "assistant",
            "text": text,
        })
    return result
