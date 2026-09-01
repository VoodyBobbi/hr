#!/bin/sh
set -e

PORT="${PORT:-8000}"

# Тот же порядок действий, что и при локальном запуске через run_all.py
# (см. этот файл про подробности) — только здесь оформлено как отдельные
# шаги shell-скрипта, потому что в Docker-режиме сайт и бот запускаются как
# два отдельных процесса, а не через один run_all.py.

echo "Проверяю ключ шифрования..."
if [ -z "$ENCRYPTION_KEY" ]; then
  echo "======================================================================"
  echo "ОШИБКА: ключ шифрования (ENCRYPTION_KEY) не задан в .env."
  echo "Без него программа не сможет сохранять анкеты кандидатов."
  echo "Сгенерируйте ключ (например, командой 'python setup.py' на компьютере"
  echo "администратора вне Docker) и добавьте его в .env перед запуском."
  echo "======================================================================"
  exit 1
fi

echo "Проверяю актуальность поискового индекса..."
python -m backend.build_index

echo "Проверяю срок хранения логов..."
python -c "from backend.logger import purge_old_logs; purge_old_logs()"

# Telegram-бот — опциональный канал. Если секрет TELEGRAM_BOT_TOKEN не задан
# в Environment Variables сервиса, просто пропускаем запуск бота (сайт работает как обычно).
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
  echo "TELEGRAM_BOT_TOKEN задан — запускаю Telegram-бота в фоне"
  python -m backend.telegram_bot &
else
  echo "TELEGRAM_BOT_TOKEN не задан — Telegram-бот не запускается, поднимаю только сайт"
fi

exec uvicorn backend.app:app --host 0.0.0.0 --port "$PORT"
