"""Защита от навязанной анкеты.

GigaChat ставит маркер START_ANKETA слишком охотно. Код обязан принимать
его только при явном согласии кандидата."""
import pytest


@pytest.mark.parametrize("text", [
    "Какие есть варианты",
    "Какие есть варианты?",
    "а какие есть?",
    "сколько платят?",
    "расскажите про данные",       # ловилось на «\bда» внутри «данные»
    "это далеко от Москвы",        # «\bда» внутри «далеко»
    "а даже так",                  # «\bда» внутри «даже»
    "окно приёма заявок закрыто",  # «\bок» внутри «окно»
    "около 100 тысяч",             # «\bок» внутри «около»
    "какая готовность нужна",
])
def test_vopros_ne_zapuskaet_anketu(bot, text):
    assert not bot.accepts_anketa(text)


@pytest.mark.parametrize("text", [
    "да", "ага", "угу", "да, давайте", "хочу оставить заявку",
    "хочу заполнить анкету", "готов", "поехали", "ок", "конечно",
])
def test_yavnoe_soglasie_zapuskaet_anketu(bot, text):
    assert bot.accepts_anketa(text)


@pytest.mark.parametrize("text", [
    "Заполнить анкету",
    "хочу оставить заявку",
    "анкета",
    "хочу работать у вас",
])
def test_anketa_zapuskaetsya_bez_neyroseti(bot, text):
    """Кнопка и прямые просьбы распознаются кодом — ноль токенов.

    Раньше запуск зависел от того, поставит ли модель служебный маркер. В
    боевом тесте кандидат с опечаткой застрял и начал анкету только с
    третьей попытки."""
    assert bot.anketa.is_anketa_request(text)
    reply = bot.say(text)
    assert reply is not None and "152-ФЗ" in reply


@pytest.mark.parametrize("text", [
    "сколько платят?",
    "какие вакансии",
    "где обучение",
])
def test_obychnyy_vopros_ne_zapuskaet_anketu(bot, text):
    assert not bot.anketa.is_anketa_request(text)
