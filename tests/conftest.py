"""
Общая обвязка для тестов.

Тяжёлые зависимости (модель эмбеддингов, GigaChat, faiss-индекс) подменяются
заглушками: тесты проверяют ЛОГИКУ анкеты и валидацию, а не качество поиска.
Поиск без настоящей модели проверить всё равно нельзя, а стейт-машина от
неё не зависит вовсе.
"""
import os
import re
import sys
import types

import numpy as np
import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stubs():
    """Заглушки вместо sentence-transformers, gigachat и httpx."""
    def fake_encode(texts, convert_to_numpy=True, normalize_embeddings=False):
        vectors = np.zeros((len(texts), 384), dtype="float32")
        vectors[:, 0] = 1.0
        return vectors

    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = lambda name: types.SimpleNamespace(encode=fake_encode)
    sys.modules.setdefault("sentence_transformers", st)

    giga = types.ModuleType("gigachat")
    giga.GigaChat = object
    ctx = types.ModuleType("gigachat.context")
    ctx.session_id_cvar = types.SimpleNamespace(set=lambda v: None)
    exc = types.ModuleType("gigachat.exceptions")
    for name in ("AuthenticationError", "GigaChatException", "RateLimitError", "ResponseError"):
        setattr(exc, name, type(name, (Exception,), {}))
    models = types.ModuleType("gigachat.models")
    models.Chat = object
    models.Messages = object
    models.MessagesRole = types.SimpleNamespace(SYSTEM="system", USER="user", ASSISTANT="assistant")
    for name, module in (("gigachat", giga), ("gigachat.context", ctx),
                         ("gigachat.exceptions", exc), ("gigachat.models", models)):
        sys.modules.setdefault(name, module)

    http = types.ModuleType("httpx")
    http.HTTPError = Exception
    sys.modules.setdefault("httpx", http)


@pytest.fixture(autouse=True)
def isolated_data(monkeypatch, tmp_path):
    """Каждый тест работает со СВОЕЙ пустой папкой данных и своим ключом.

    Без изоляции тесты видели бы карточки друг друга, а при неудачном
    стечении — настоящие анкеты разработчика."""
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("HR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "stub")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:8000")
    _install_stubs()

    for name in [m for m in list(sys.modules) if m.startswith("backend")]:
        del sys.modules[name]
    yield tmp_path


@pytest.fixture
def bot():
    """Диалог с ботом: bot.say("текст") -> ответ анкетной ветки (или None).

    Загружается только логика анкеты из assistant.py, без запуска сервера."""
    from backend import anketa, candidates, notifications

    sent = []
    notifications.send_hr_notification = lambda text, buttons=None: sent.append((text, buttons))

    source_code = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "backend", "assistant.py"),
        encoding="utf-8",
    ).read()
    from backend.validators import FieldValidationError
    namespace = {
        "candidates": candidates, "anketa": anketa, "notifications": notifications,
        "FieldValidationError": FieldValidationError, "re": re,
    }
    chunk = source_code[source_code.index("def _handle_anketa_turn("):
                        source_code.index("# Служебная подсказка о заполненных полях")]
    exec(compile(chunk, "assistant_part", "exec"), namespace)
    guard = source_code[source_code.index("_ANKETA_INTENT = ("):
                        source_code.index("def _format_kb_context(")]
    exec(compile(guard, "assistant_guard", "exec"), namespace)

    class Bot:
        def __init__(self):
            self.last = ""
            self.ext = "test-session"
            self.sent = sent
            self.candidates = candidates
            self.anketa = anketa
            self.accepts_anketa = namespace["_user_really_asked_for_anketa"]

        def say(self, text, start_requested=False):
            reply = namespace["_handle_anketa_turn"](
                "site", self.ext, text, self.last,
                anketa_start_requested=start_requested,
            )
            if reply is not None:
                self.last = reply
            return reply

        def start_anketa(self):
            """Доводит диалог до первого вопроса анкеты."""
            consent = self.say("хочу оставить заявку", start_requested=True)
            # Как в get_answer: текст модели склеивается с согласием.
            self.last = "Отлично, тогда начнём.\n\n" + consent
            return self.say("да")

        @property
        def card_id(self):
            return candidates.find_candidate("site", self.ext)

        def card(self):
            return candidates.get_card(self.card_id)

    return Bot()


FULL_ANSWERS = [
    "Иванов", "Иван", "Нет отчества", "15.03.1995", "89991234567", "@ivanov",
    "4510", "123456", "ОУФМС России", "12.05.2015", "740-021",
    "г. Москва, ул. Ленина, 1", "Совпадает с регистрацией",
    "12345678901", "нет", "Россия", "Москва", "Промышленный альпинист",
    "5 лет на высоте", "ООО Стройка", "Среднее специальное, допуск на высоту",
    "178", "75", "50", "42", "нет", "нет", "нет", "с 1 октября",
]
