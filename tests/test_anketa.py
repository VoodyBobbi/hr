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


def test_vopros_na_shage_soglasiya_uhodit_v_obychnyy_dialog(bot):
    """Вопрос вместо «да» или «нет» должен получить ответ, а не повтор
    согласия. Раньше человек спрашивал «зарплата?» и получал юридический
    текст целиком — и так на каждый вопрос."""
    consent = bot.say("хочу оставить заявку", start_requested=True)
    bot.last = "Отлично.\n\n" + consent

    assert bot.say("а сколько платят?") is None, "вопрос должен уйти в GigaChat"
    assert bot.card_id is None, "карточка не должна заводиться до согласия"


def test_neponyatnyy_otvet_pereprashivaet(bot):
    """А вот бессмысленный ответ, который не вопрос, — переспрашиваем."""
    consent = bot.say("хочу оставить заявку", start_requested=True)
    bot.last = "Отлично.\n\n" + consent
    assert "Не понял" in bot.say("хм")
    assert bot.card_id is None


def test_net_davayte_pozzhe_eto_otkaz(bot):
    """152-ФЗ: «Нет, давайте позже» — отказ, а не согласие."""
    consent = bot.say("хочу оставить заявку", start_requested=True)
    bot.last = "Отлично.\n\n" + consent
    assert "отложим" in bot.say("Нет, давайте позже")
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


def test_zamok_aktivnogo_dialoga_ne_vytesnyaetsya(bot):
    """Замок диалога, которым пользуются, не должен вылетать при переполнении.

    Если вытеснять по порядку создания, замок кандидата, заполняющего
    анкету целый час, вылетит первым. На его месте появится новый объект, и
    два сообщения одного человека обработаются параллельно — с гонкой за
    файл истории и перепутанным порядком реплик."""
    import re
    import threading
    from collections import OrderedDict

    source = open("backend/assistant.py", encoding="utf-8").read()
    chunk = source[source.index("_MAX_SESSION_LOCKS"):source.index("def get_answer(")]
    ns = {"threading": threading, "OrderedDict": OrderedDict, "re": re,
          "_session_locks": OrderedDict(), "_session_locks_guard": threading.Lock()}
    exec(compile(chunk, "locks", "exec"), ns)
    session_lock = ns["_session_lock"]
    limit = ns["_MAX_SESSION_LOCKS"]

    active = session_lock("site", "активный")
    for i in range(limit):
        session_lock("site", f"случайный-{i}")
        session_lock("site", "активный")          # им продолжают пользоваться

    assert session_lock("site", "активный") is active, "активный замок подменили"


def test_vopros_posredi_ankety_predlagaet_pauzu(bot):
    """Вопрос вместо ответа на пункт — предложить выбор, а не молча
    повторять вопрос анкеты. Раньше выглядело так, будто вопрос человека
    проигнорировали."""
    bot.start_anketa()
    bot.say("Иванов")

    reply = bot.say("а сколько платят?")
    assert "вопрос, а не ответ" in reply
    assert bot.candidates.get_stage(bot.card_id) == bot.anketa.STAGE_PAUSE_ASK


def test_pauza_i_vozvrat_k_ankete(bot):
    bot.start_anketa()
    bot.say("Иванов")
    bot.say("а сколько платят?")

    assert "отложил" in bot.say("да")
    assert bot.candidates.get_stage(bot.card_id) == bot.anketa.STAGE_PAUSED

    # На паузе вопросы уходят в обычный диалог.
    assert bot.say("а что по вахте?") is None

    reply = bot.say("продолжить")
    assert "Возвращаемся к анкете" in reply
    assert bot.candidates.get_stage(bot.card_id) == bot.anketa.STAGE_FIELDS


def test_otkaz_ot_pauzy_prodolzhaet_anketu(bot):
    bot.start_anketa()
    bot.say("Иванов")
    bot.say("а сколько платят?")

    reply = bot.say("нет")
    assert "Как вас зовут" in reply, "должен повторить текущий вопрос"
    assert bot.candidates.get_stage(bot.card_id) == bot.anketa.STAGE_FIELDS
