"""Проверка живости и устойчивость к неверному ключу шифрования.

Сценарий, ради которого это всё: администратор переносит .env на сервер и
теряет символ в ключе. Раньше сервер запускался, /health отвечал «ок», а
первый кандидат получал ошибку вместо ответа."""
import os

import pytest


def test_health_soobschaet_ob_ispravnosti():
    """Проверка живости должна смотреть на ключ и индекс, а не отвечать «ок».

    Функция вызывается напрямую, без поднятия сервера. Тестовый клиент
    FastAPI тянет за собой httpx, а его подменяют заглушкой другие тесты в
    этом же прогоне — и клиент падал на чужой подмене, хотя проверяемый код
    исправен. Прямой вызов от этого избавлен и проверяет ровно то, что
    нужно."""
    pytest.importorskip("fastapi")
    from backend.app import health

    class FakeResponse:
        status_code = 200

    result = health(FakeResponse())
    assert result["encryption"] is True, "ключ шифрования должен проверяться"
    assert "index" in result, "загрузку индекса тоже надо проверять"


def test_zhurnal_ne_roniaet_otvet_pri_bitom_kluche(monkeypatch, capsys):
    """Потерянная строка журнала — неприятность. Потерянный ответ — потерянный
    кандидат."""
    from backend import logger

    def broken(_value):
        raise ValueError("ключ сломан")

    monkeypatch.setattr(logger, "_encrypt_field", broken)

    # Не должно бросить наружу.
    logger.log_interaction("site", "u1", "вопрос", "ответ", 10, "ok", "")

    assert "Не удалось записать" in capsys.readouterr().out


def test_proverka_kluchа_pri_starte_lovit_opechatku(monkeypatch):
    """run_all останавливает запуск, если ключ есть, но нерабочий."""
    import importlib
    import sys

    monkeypatch.setenv("ENCRYPTION_KEY", "это-точно-не-ключ")
    for name in [m for m in list(sys.modules) if m.startswith("backend")]:
        del sys.modules[name]

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_all = importlib.import_module("run_all")

    try:
        run_all._check_encryption_key()
    except SystemExit as e:
        assert e.code == 1
    else:
        raise AssertionError("запуск обязан остановиться при неверном ключе")
