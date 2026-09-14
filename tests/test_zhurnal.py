"""Единый журнал: разговоры и системные события в одном файле.

Главное требование — панель мониторинга должна читать события БЕЗ ключа
шифрования, а переписка кандидатов при этом оставаться закрытой."""
import csv
import io
import os

from conftest import FULL_ANSWERS


def _rows():
    from backend import logger
    with io.open(logger.LOG_PATH, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_sobytie_i_dialog_v_odnom_fayle():
    from backend import logger
    logger.log_event(logger.EVENT_START, "Проект запущен.")
    logger.log_interaction("site", "u1", "вопрос", "ответ", 12, "ok", "")

    rows = _rows()
    events = {r["Событие"] for r in rows}
    assert logger.EVENT_START in events
    assert logger.EVENT_DIALOG in events
    assert len({r["Событие"] for r in rows}) >= 2, "оба типа в одном файле"


def test_kommentariy_sobytiya_ne_zashifrovan():
    """Панель мониторинга читает события без ключа."""
    from backend import logger
    logger.log_event(logger.EVENT_NOTIFY, "Уведомление HR отправлено.")
    raw = io.open(logger.LOG_PATH, encoding="utf-8-sig").read()
    assert "Уведомление HR отправлено." in raw


def test_tekst_kandidata_v_sobytii_zashifrovan():
    """Всё, что написал человек, шифруется даже внутри события."""
    from backend import logger
    secret = "мой паспорт 4510 123456"
    logger.log_event(
        logger.EVENT_MARKER, "Маркер отклонён.",
        status=logger.STATUS_WARN, details=secret,
    )
    raw = io.open(logger.LOG_PATH, encoding="utf-8-sig").read()
    assert secret not in raw, "текст кандидата не должен лежать открытым"
    assert "Маркер отклонён." in raw


def test_udalenie_ostavlyaet_sled(bot):
    """152-ФЗ: по журналу должно быть видно, что и когда удалили."""
    from backend import logger
    bot.start_anketa()
    for answer in FULL_ANSWERS:
        bot.say(answer)
    bot.say("да, всё верно")
    bot.say("удали мои данные")
    bot.say("да, удали")

    deletes = [r for r in _rows() if r["Событие"] == logger.EVENT_DELETE]
    assert deletes, "событие удаления должно попасть в журнал"
    assert "Удалены данные кандидата" in deletes[-1]["Комментарий"]


def test_uvedomlenie_bez_nastroek_pishet_oshibku(monkeypatch):
    """Самый коварный случай: HR ничего не получил и не узнал бы об этом.

    Фикстура bot здесь НЕ используется намеренно: она подменяет
    send_hr_notification заглушкой, и тест проверял бы заглушку, а не
    настоящую отправку."""
    from backend import logger, notifications
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_HR_GROUP_CHAT_ID", raising=False)

    notifications.send_hr_notification("тест")

    notify = [r for r in _rows() if r["Событие"] == logger.EVENT_NOTIFY]
    assert notify, "неотправленное уведомление обязано попасть в журнал"
    assert notify[-1]["Статус"] == logger.STATUS_ERROR


def test_staryy_zhurnal_migriruet():
    """Файл без колонки «Событие» не должен ломать запись."""
    from backend import logger
    os.makedirs(os.path.dirname(logger.LOG_PATH), exist_ok=True)
    with io.open(logger.LOG_PATH, "w", encoding="utf-8-sig", newline="") as f:
        f.write("Дата и время,Источник,ID пользователя,Вопрос,Ответ\n")
        f.write("2026-01-01 10:00:00,site,old,x,y\n")

    logger.log_event(logger.EVENT_START, "После миграции.")
    rows = _rows()
    assert "Событие" in rows[0], "заголовок должен обновиться"
    assert len(rows) == 2, "старая строка не должна потеряться"
