"""Балльная защита от спама.

Главное требование: она НЕ должна мешать обычному кандидату. Прежняя
версия блокировала человека на третьем поле анкеты."""
import pytest


@pytest.fixture
def rl():
    from backend import rate_limiting
    rate_limiting._state.clear()
    return rate_limiting


def test_ballы_zatuhayut_so_vremenem(rl):
    """30 сообщений раз в полминуты — это обычный разговор, не спам."""
    key = "session:normal"
    for i in range(30):
        assert not rl.is_blocked("site", key)[0]
        rl.record_message("site", key, f"вопрос {i}")
        rl._state["site:" + key]["last_time"] -= 30.0
    assert rl._state["site:" + key]["points"] == 0


def test_otvety_v_ankete_ne_blokiruyutsya(rl):
    """Ответ раз в три секунды — нормальный темп заполнения формы."""
    key = "session:anketa"
    for answer in ["Иванов", "Иван", "нет", "нет", "нет", "42", "178"]:
        assert not rl.is_blocked("site", key)[0]
        rl.record_message("site", key, answer)
        rl._state["site:" + key]["last_time"] -= 3.0


def test_spam_blokiruetsya(rl):
    """Одно и то же сообщение без пауз — блокировка."""
    key = "session:spam"
    for _ in range(15):
        if rl.is_blocked("site", key)[0]:
            return
        rl.record_message("site", key, "AAAA")
    pytest.fail("спамер не заблокирован")


def test_schetchik_vyklyuchen_vo_vremya_ankety(bot):
    """Шаги анкеты не обращаются к нейросети — защищать нечего."""
    from backend import rate_limiting
    assert not rate_limiting.in_anketa("site", bot.ext)
    bot.start_anketa()
    assert rate_limiting.in_anketa("site", bot.ext)
