"""
Проверка настройки уведомлений HR в Telegram.

    python -m scripts.check_telegram

Ничего не меняет. Последовательно проверяет всё, что нужно для доставки
уведомления о новой анкете, и на каждом шаге говорит, что именно не так.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

API = "https://api.telegram.org/bot{token}/{method}"


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_HR_GROUP_CHAT_ID", "").strip()

    print("1. Переменные в .env")
    if not token:
        print("   TELEGRAM_BOT_TOKEN — ПУСТО.")
        print("   Получите токен у @BotFather и впишите его в .env.")
        return
    print(f"   TELEGRAM_BOT_TOKEN — задан ({token[:10]}...)")

    if not chat_id:
        print("   TELEGRAM_HR_GROUP_CHAT_ID — ПУСТО.")
        print()
        print("   Это и есть причина, по которой уведомления не приходят.")
        print("   Впишите в .env строку с номером вашей группы, например:")
        print("       TELEGRAM_HR_GROUP_CHAT_ID=-1003869481165")
        print()
        _print_how_to_find_chat_id(token)
        return
    print(f"   TELEGRAM_HR_GROUP_CHAT_ID — задан ({chat_id})")

    print()
    print("2. Токен рабочий?")
    try:
        r = httpx.get(API.format(token=token, method="getMe"), timeout=10.0)
    except httpx.HTTPError as e:
        print(f"   Нет связи с Telegram: {e}")
        return
    if r.status_code != 200:
        print(f"   Telegram отклонил токен ({r.status_code}). Проверьте его в .env.")
        return
    bot = r.json().get("result", {})
    print(f"   Да, бот @{bot.get('username')} ({bot.get('first_name')})")

    print()
    print("3. Бот видит группу?")
    try:
        r = httpx.get(
            API.format(token=token, method="getChat"),
            params={"chat_id": chat_id}, timeout=10.0,
        )
    except httpx.HTTPError as e:
        print(f"   Нет связи с Telegram: {e}")
        return
    if r.status_code != 200:
        print(f"   НЕТ. Telegram ответил: {r.text}")
        print()
        print("   Возможные причины:")
        print("   - бот не добавлен в группу участником;")
        print("   - номер группы неверный. Попробуйте вариант с префиксом -100:")
        if not chat_id.startswith("-100"):
            print(f"       TELEGRAM_HR_GROUP_CHAT_ID=-100{chat_id.lstrip('-')}")
        print()
        _print_how_to_find_chat_id(token)
        return
    chat = r.json().get("result", {})
    print(f"   Да, это «{chat.get('title')}» (тип: {chat.get('type')})")

    print()
    print("4. Пробная отправка")
    try:
        r = httpx.post(
            API.format(token=token, method="sendMessage"),
            json={"chat_id": chat_id,
                  "text": "Проверка связи. Если вы видите это сообщение — "
                          "уведомления о новых анкетах будут приходить сюда."},
            timeout=10.0,
        )
    except httpx.HTTPError as e:
        print(f"   Нет связи с Telegram: {e}")
        return
    if r.status_code != 200:
        print(f"   НЕ отправлено. Telegram ответил: {r.text}")
        print("   Если написано 'not enough rights' — сделайте бота "
              "администратором группы.")
        return

    print("   Отправлено. Загляните в группу — сообщение должно быть там.")
    print()
    print("Всё настроено верно.")


def _print_how_to_find_chat_id(token: str):
    print("   Как узнать номер группы:")
    print("     1. Добавьте бота в группу и сделайте администратором.")
    print("     2. Напишите в группе любое сообщение.")
    print("     3. Откройте в браузере:")
    print(f"        https://api.telegram.org/bot{token}/getUpdates")
    print('     4. Найдите "chat":{"id":-100...} — это нужное число.')


if __name__ == "__main__":
    main()
