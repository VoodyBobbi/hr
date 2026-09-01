"""
Разовая миграция: если где-то уже накопился СТАРЫЙ незашифрованный
candidates.csv (например, скачан с прежнего сервера до перехода на локальный
запуск) — эта команда зашифрует его и положит в новое хранилище
(по умолчанию HR/candidates/candidates.csv внутри папки проекта, либо
HR_DATA_DIR/candidates/candidates.csv, если задан свой путь), откуда его
будет читать основной код.

Запуск из корня проекта (нужен настроенный .env с ENCRYPTION_KEY):
    python -m scripts.migrate_plaintext_csv путь/к/старому/candidates.csv

Если в новом хранилище уже есть файл — команда откажется его перезаписать
(на всякий случай, чтобы не потерять данные), удалите/переименуйте старый
вручную, если точно хотите заменить.
"""
import argparse
import os
import sys

from backend import paths
from backend.crypto_utils import encrypt_bytes


def main():
    parser = argparse.ArgumentParser(description="Зашифровать старый plaintext candidates.csv в новое хранилище.")
    parser.add_argument("source_csv", help="Путь к старому незашифрованному candidates.csv")
    args = parser.parse_args()

    if not os.path.exists(args.source_csv):
        print(f"Файл не найден: {args.source_csv}")
        sys.exit(1)

    target = os.path.join(paths.CANDIDATS_DIR, "candidates.csv")
    if os.path.exists(target):
        print(f"В новом хранилище уже есть файл: {target}")
        print("Ничего не делаю, чтобы не перезаписать существующие данные. "
              "Переименуйте/удалите его вручную, если точно хотите заменить.")
        sys.exit(1)

    with open(args.source_csv, "rb") as f:
        plaintext = f.read()

    # Простая проверка, что это похоже на обычный (не уже зашифрованный) CSV —
    # человекочитаемый заголовок таблицы, а не бинарный Fernet-токен.
    if not plaintext.lstrip(b"\xef\xbb\xbf").startswith(("Поле".encode("utf-8"))):
        print("Файл не похож на исходный candidates.csv (нет заголовка 'Поле' в начале) — "
              "проверьте, тот ли файл указан. Ничего не сделано.")
        sys.exit(1)

    ciphertext = encrypt_bytes(plaintext)
    with open(target, "wb") as f:
        f.write(ciphertext)

    print(f"Готово: {args.source_csv} зашифрован и сохранён в {target}")
    print("Проверьте результат обычным запуском приложения — карточки кандидатов должны читаться как раньше.")


if __name__ == "__main__":
    main()
