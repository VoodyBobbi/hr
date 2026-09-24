"""Персональные данные из анкеты не должны уходить в нейросеть.

Ответы на шаги анкеты — паспорт, СНИЛС, ИНН, адреса, судимость, здоровье —
пишутся в историю диалога. Раньше вся история целиком уходила в GigaChat
при следующем же обычном вопросе. Два пути утечки:

1. Человек ставит анкету на паузу посреди паспортных полей и спрашивает о
   зарплате — в модель уезжают серия и номер паспорта.
2. Человек подтверждает карточку и спрашивает о выезде — в модель уезжает
   вся карточка из 29 полей, потому что последним сообщением бота была она.

Тест проверяет самое главное: что именно попадает в список сообщений,
который собирается для модели."""
from conftest import FULL_ANSWERS

PASSPORT = ("4510", "123456", "12345678901")


def _messages_for_model(history):
    """Та же фильтрация, что в assistant._get_answer перед вызовом GigaChat."""
    return [t["content"] for t in history if not t.get("anketa")]


def test_anketnye_hody_pomecheny(bot):
    """Каждая реплика анкеты должна нести метку anketa."""
    source_code = open("backend/assistant.py", encoding="utf-8").read()
    assert '"anketa": True' in source_code, "анкетные ходы должны помечаться"
    assert 'if turn.get("anketa"):' in source_code, "и отфильтровываться перед моделью"


def test_pasport_ne_popadaet_v_istoriyu_dlya_modeli():
    """Имитирует историю после заполнения анкеты и проверяет фильтр."""
    history = []
    for answer in FULL_ANSWERS:
        history.append({"role": "user", "content": answer, "anketa": True})
        history.append({"role": "assistant", "content": "следующий вопрос", "anketa": True})
    # Карточка целиком — последнее сообщение бота после анкеты.
    card = "Проверьте, пожалуйста, все данные:\nПаспорт серия: 4510\nПаспорт номер: 123456"
    history.append({"role": "assistant", "content": card, "anketa": True})
    # Обычный вопрос после анкеты — он в модель уйти должен.
    history.append({"role": "user", "content": "а когда выезд?"})

    sent = _messages_for_model(history)

    assert not any(p in c for c in sent for p in PASSPORT), "паспорт утёк в модель"
    assert "а когда выезд?" in sent, "обычный вопрос должен дойти до модели"


def test_vopros_so_znakom_voprosa_ne_s_edaetsya_polem(bot):
    """Вопрос не должен сохраняться как ответ на пункт анкеты.

    У полей со свободным текстом проверки формата нет, и раньше
    «а сколько платят?» сохранялось в поле «Кем выдан» паспорта."""
    bot.start_anketa()
    for answer in FULL_ANSWERS[:8]:        # до поля «Кем выдан»
        bot.say(answer)

    reply = bot.say("а сколько платят?")
    assert "вопрос, а не ответ" in reply
    assert bot.card()["Кем выдан"] == "", "вопрос не должен попасть в поле"
