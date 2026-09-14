"""Удаление данных по 152-ФЗ. Самая ответственная часть: если тут ошибка,
персональные данные остаются на диске вопреки прямой просьбе человека."""
import os

from conftest import FULL_ANSWERS


def _zapolnit_i_podtverdit(bot):
    bot.start_anketa()
    for answer in FULL_ANSWERS:
        bot.say(answer)
    bot.say("да, всё верно")


def test_udalenie_posle_otpravki_ankety(bot):
    """Право отозвать согласие возникает ПОСЛЕ передачи данных."""
    _zapolnit_i_podtverdit(bot)
    card_id = bot.card_id
    assert "уверены" in bot.say("удали мои данные").lower()
    assert "удалены" in bot.say("да, удали")
    assert bot.card_id is None
    assert card_id not in bot.candidates._load_table()[1]


def test_fail_istorii_ne_voskresaet(bot, isolated_data):
    """Файл переписки не должен создаваться заново сразу после удаления."""
    from backend import paths
    _zapolnit_i_podtverdit(bot)
    bot.say("удали мои данные")
    reply = bot.say("да, удали")
    assert reply == bot.anketa.DELETE_DONE_MESSAGE
    assert not os.path.exists(paths.conversation_path("site", bot.ext))


def test_povtor_prosby_ne_otmenyaet_udalenie(bot):
    """Люди перефразируют вместо подтверждения — это не отказ."""
    _zapolnit_i_podtverdit(bot)
    bot.say("удали мои данные")
    reply = bot.say("удалите мои данные пожалуйста")
    assert "уверены" in reply.lower(), "должен переспросить, а не вернуть в анкету"
    assert bot.candidates.get_stage(bot.card_id) == bot.anketa.STAGE_DELETE


def test_otkaz_ot_udaleniya_vozvraschaet_v_anketu(bot):
    _zapolnit_i_podtverdit(bot)
    bot.say("удали мои данные")
    reply = bot.say("нет, передумал")
    assert reply.startswith("Проверьте, пожалуйста")
    assert bot.card_id is not None


def test_udali_iz_rassylki_ne_zapuskaet_udalenie(bot):
    _zapolnit_i_podtverdit(bot)
    assert bot.say("удали меня из рассылки") is None
    assert bot.card_id is not None


def test_otricanie_ne_zapuskaet_udalenie(bot):
    """«Только не удали мои данные» — просьба об обратном.

    Раньше фраза содержала «удали мои данные» целиком, и человек получал
    тревожное «вы уверены, это необратимо», хотя просил не трогать."""
    for text in ("не удали мои данные",
                 "только не удаляй мои данные",
                 "пожалуйста не удаляйте мои данные"):
        assert not bot.anketa.is_delete_request(text), text


def test_obychnaya_prosba_po_prezhnemu_rabotaet(bot):
    for text in ("удали мои данные",
                 "хочу удалить свои данные",
                 "удалите мою заявку"):
        assert bot.anketa.is_delete_request(text), text
