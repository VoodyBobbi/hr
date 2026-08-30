"""
Уведомления HR-группе в Telegram о новых полностью заполненных анкетах.

Сделано отдельным лёгким модулем (прямой HTTP-запрос к Bot API через httpx),
а не через python-telegram-bot Application, по двум причинам:
1. Application — это тяжёлый объект с собственным event loop, рассчитанный
   на приём сообщений (long polling), а не просто на отправку одного
   сообщения. Поднимать его только ради одного sendMessage — избыточно.
2. Уведомление должно уходить одинаково, откуда бы ни пришла анкета — с
   сайта (backend/app.py) или из Telegram (backend/telegram_bot.py). Прямой
   HTTP-вызов к Bot API работает одинаково в обоих случаях; полноценный
   Application по устройству библиотеки привязан к процессу самого бота.

Как узнать TELEGRAM_HR_GROUP_CHAT_ID — см. README, раздел "Уведомления HR
о новых заявках".
"""
import os

import httpx

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_HR_GROUP_CHAT_ID = os.getenv("TELEGRAM_HR_GROUP_CHAT_ID")

_API_URL_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"


def hr_notifications_enabled() -> bool:
    """True, если заданы обе переменные, нужные для отправки уведомлений
    HR-группе. Если что-то не задано — уведомления просто тихо не
    отправляются (это НЕ обязательная функция: без Telegram-канала или без
    настроенной группы весь остальной бот должен продолжать работать)."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_HR_GROUP_CHAT_ID)


def send_hr_notification(text: str) -> None:
    """Отправляет текстовое сообщение в HR-группу. Никогда не бросает
    исключение наружу — сбой уведомления не должен ронять сам ответ
    кандидату (это дополнительная, не критичная для работы бота функция)."""
    if not hr_notifications_enabled():
        return

    url = _API_URL_TEMPLATE.format(token=TELEGRAM_BOT_TOKEN)
    try:
        response = httpx.post(
            url,
            json={
                "chat_id": TELEGRAM_HR_GROUP_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            },
            timeout=10.0,
        )
        if response.status_code != 200:
            print(
                f"[notifications] Telegram Bot API вернул {response.status_code} "
                f"при отправке уведомления HR-группе: {response.text}"
            )
    except httpx.HTTPError as e:
        print(f"[notifications] Не удалось отправить уведомление HR-группе: {e}")


def format_new_candidate_notification(card: dict) -> str:
    """Формирует текст уведомления о новой полностью заполненной анкете.

    Намеренно НЕ включает чувствительные персональные данные (паспорт,
    СНИЛС, ИНН, адреса) — только то, что нужно HR, чтобы узнать заявку и
    решить, открывать ли карточку кандидата. Полная анкета целиком уже
    надёжно лежит в зашифрованном candidates.csv — дублировать её в
    групповой чат Telegram (менее защищённый канал, доступный всем
    участникам группы) было бы шагом назад по защите данных, а не вперёд.
    """
    full_name = " ".join(
        part for part in (card.get("Фамилия"), card.get("Имя"), card.get("Отчество")) if part
    ).strip() or "(ФИО не указано)"

    source_label = {"site": "сайт", "telegram": "Telegram"}.get(card.get("Источник"), card.get("Источник", "—"))
    law_ack = card.get("Факт ознакомления с ЗАКОНОМ", "").strip()
    law_status = "да, " + law_ack if law_ack else "нет данных"

    return (
        f"<b>Новая заявка от кандидата</b>\n"
        f"ФИО: {full_name}\n"
        f"Вакансия: {card.get('Желаемая должность', '—')}\n"
        f"Телефон: {card.get('Телефон', '—')}\n"
        f"Telegram: {card.get('Telegram', '—')}\n"
        f"Канал обращения: {source_label}\n"
        f"ID кандидата в таблице: {card.get('ID кандидата', '—')}\n"
        f"Ознакомлен(а) с 152-ФЗ: {law_status}\n"
        f"\nПолная анкета — в зашифрованной таблице candidates.csv (id {card.get('ID кандидата', '—')})."
    )
