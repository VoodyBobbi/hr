"""Стейт-машина анкеты: согласие, сбор полей, правка, подтверждение."""
from conftest import FULL_ANSWERS


def test_vopros_do_ankety_ne_zavodit_kartochku(bot):
    """Обычный вопрос уходит в GigaChat и не создаёт запись в таблице ПД."""
    assert bot.say("какие есть вакансии?") is None
    assert bot.card_id is None


def test_soglasie_zavodit_kartochku(bot):
    question = bot.start_anketa()
    assert "фамилию" in question.lower()
    assert bot.card_id is not None


def test_neponyatnyy_otvet_na_soglasie_pereprashivaet(bot):
    consent = bot.say("хочу оставить заявку", start_requested=True)
    bot.last = "Отлично.\n\n" + consent
    assert "Не понял" in bot.say("а долго это?")
    assert bot.card_id is None, "карточка не должна заводиться до согласия"


def test_net_davayte_pozzhe_eto_otkaz(bot):
    """152-ФЗ: «Нет, давайте позже» — отказ, а не согласие."""
    consent = bot.say("хочу оставить заявку", start_requested=True)
    bot.last = "Отлично.\n\n" + consent
    assert "не будем" in bot.say("Нет, давайте позже")
    assert bot.card_id is None


def test_polnoe_zapolnenie(bot):
    bot.start_anketa()
    for answer in FULL_ANSWERS:
        bot.say(answer)
    assert bot.candidates.is_card_complete(bot.card_id)
    assert bot.last.startswith("Проверьте, пожалуйста")


def test_telefon_normalizuetsya(bot):
    bot.start_anketa()
    for answer in FULL_ANSWERS:
        bot.say(answer)
    assert bot.card()["Телефон"] == "+79991234567"


def test_pravka_polya_sohranyaetsya(bot):
    bot.start_anketa()
    for answer in FULL_ANSWERS:
        bot.say(answer)
    reply = bot.say("Телефон 89998887766")
    assert "Исправил" in reply
    assert bot.card()["Телефон"] == "+79998887766"


def test_da_no_telefon_eto_pravka_a_ne_podtverzhdenie(bot):
    """«Да, но телефон ...» не должно закрывать анкету со старым номером."""
    bot.start_anketa()
    for answer in FULL_ANSWERS:
        bot.say(answer)
    bot.say("да, но телефон 89998887766")
    assert bot.card()["Телефон"] == "+79998887766"
    assert bot.candidates.get_stage(bot.card_id) == bot.anketa.STAGE_CONFIRM


def test_podtverzhdenie_otpravlyaet_uvedomlenie(bot):
    bot.start_anketa()
    for answer in FULL_ANSWERS:
        bot.say(answer)
    assert "принята" in bot.say("да, всё верно")
    assert bot.sent, "уведомление HR должно уйти"
    text, buttons = bot.sent[-1]
    assert buttons, "к уведомлению должна прилагаться кнопка"
    assert "4510" not in text, "паспорт не должен попадать в ленту группы"


def test_posle_ankety_obychnyy_dialog(bot):
    bot.start_anketa()
    for answer in FULL_ANSWERS:
        bot.say(answer)
    bot.say("да, всё верно")
    assert bot.say("а когда выезд?") is None


def test_novoe_pole_ne_teryaetsya_v_starom_fayle(bot):
    """Поле, добавленное в новой версии, должно дописаться в таблицу.

    Из-за потери такого поля «Версия анкеты» всегда оставалась пустой,
    любая карточка считалась устаревшей, и кандидата выбивало из анкеты
    после первого же ответа. Заполнить её не мог никто."""
    from backend import candidates
    old_fields = [f for f in candidates.FIELD_ORDER if f != "Версия анкеты"]
    candidates._save_table(old_fields, {"1": {f: "" for f in old_fields}})

    assert "Версия анкеты" in candidates._load_table()[0]

    bot.start_anketa()
    assert not candidates.is_outdated(bot.card_id)


def test_iz_zavershennoy_ankety_mozhno_nachat_zanovo(bot):
    """Человек уже подал заявку и хочет подать ещё одну.

    Раньше здесь была ловушка: бот предлагал написать «хочу оставить
    заявку», человек писал — и не получал ничего, кроме «Пустой ответ
    сервера». Выбраться было невозможно."""
    from conftest import FULL_ANSWERS
    bot.start_anketa()
    for answer in FULL_ANSWERS:
        bot.say(answer)
    bot.say("да, всё верно")
    assert bot.candidates.get_stage(bot.card_id) == bot.anketa.STAGE_DONE

    reply = bot.say("хочу оставить заявку", start_requested=True)
    assert reply is not None, "бот обязан ответить, а не молчать"
    assert "152-ФЗ" in reply, "должен показать согласие и начать заново"
