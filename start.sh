#!/bin/sh
set -e

# Telegram-бот — опциональный канал. Если секрет TELEGRAM_BOT_TOKEN не задан
# в настройках Space, просто пропускаем запуск бота (сайт при этом работает как обычно).
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
  echo "TELEGRAM_BOT_TOKEN задан — запускаю Telegram-бота в фоне"
  python -m backend.telegram_bot &
else
  echo "TELEGRAM_BOT_TOKEN не задан — Telegram-бот не запускается, поднимаю только сайт"
fi

exec uvicorn backend.app:app --host 0.0.0.0 --port 7860
