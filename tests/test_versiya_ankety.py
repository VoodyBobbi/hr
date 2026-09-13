"""Версионирование анкеты.

Карточка, начатая на прежнем наборе полей, после обновления несовместима
с новым: вопросы и сохранённые ответы разъезжаются. Раньше это проявлялось
молча — анкета сбивалась на середине."""


def test_novaya_kartochka_imeet_tekuschuyu_versiyu(bot):
    from backend import candidates
    bot.start_anketa()
    assert bot.card()["Версия анкеты"] == candidates.ANKETA_VERSION
    assert not candidates.is_outdated(bot.card_id)


def test_staraya_kartochka_priznaetsya_ustarevshey(bot):
    from backend import candidates
    bot.start_anketa()
    candidates.set_field(bot.card_id, "Версия анкеты", "1")
    assert candidates.is_outdated(bot.card_id)


def test_ustarevshaya_anketa_predlagaet_nachat_zanovo(bot):
    from backend import candidates
    bot.start_anketa()
    bot.say("Иванов")
    candidates.set_field(bot.card_id, "Версия анкеты", "1")
    reply = bot.say("Иван")
    assert "обновилась" in reply
    assert "заново" in reply


def test_udalenie_rabotaet_dazhe_na_ustarevshey(bot):
    """Право на удаление не должно зависеть от версии формы."""
    from backend import candidates
    bot.start_anketa()
    candidates.set_field(bot.card_id, "Версия анкеты", "1")
    assert "уверены" in bot.say("удали мои данные").lower()
    assert "удалены" in bot.say("да, удали")
    assert bot.card_id is None
