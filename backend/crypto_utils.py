"""
Шифрование файла с анкетами кандидатов.

Схема: симметричное шифрование всего файла целиком (Fernet, AES-128-CBC +
HMAC внутри). Ключ хранится ОТДЕЛЬНО от файла — в переменной окружения
ENCRYPTION_KEY (.env или окружение ОС), а не в коде и не рядом с самим
файлом на диске. Это тот же принцип, что уже применяется к
GIGACHAT_CREDENTIALS в этом проекте.

Как получить ключ для .env:
    python -m backend.crypto_utils

Команда сгенерирует новый случайный ключ и напечатает готовую строку
для .env. Ключ нужно сгенерировать один раз и сохранить — потеря ключа
означает потерю доступа к данным в candidates.csv.
"""
import os

from cryptography.fernet import Fernet, InvalidToken

_ENV_VAR = "ENCRYPTION_KEY"


class EncryptionKeyMissing(RuntimeError):
    pass


class EncryptionKeyInvalid(RuntimeError):
    pass


def _get_key() -> bytes:
    key = os.environ.get(_ENV_VAR)
    if not key:
        raise EncryptionKeyMissing(
            f"Переменная окружения {_ENV_VAR} не задана. Сгенерируйте ключ командой "
            f"'python -m backend.crypto_utils' и добавьте строку ENCRYPTION_KEY=... в .env."
        )
    try:
        return key.encode("utf-8")
    except UnicodeEncodeError as e:
        raise EncryptionKeyInvalid(f"{_ENV_VAR} содержит недопустимые символы.") from e


def _fernet() -> Fernet:
    key = _get_key()
    try:
        return Fernet(key)
    except (ValueError, TypeError) as e:
        raise EncryptionKeyInvalid(
            f"{_ENV_VAR} имеет неверный формат — это должен быть ключ, "
            f"сгенерированный 'python -m backend.crypto_utils', а не произвольная строка."
        ) from e


def encrypt_bytes(plaintext: bytes) -> bytes:
    return _fernet().encrypt(plaintext)


def decrypt_bytes(ciphertext: bytes) -> bytes:
    try:
        return _fernet().decrypt(ciphertext)
    except InvalidToken as e:
        raise EncryptionKeyInvalid(
            "Не удалось расшифровать файл текущим ENCRYPTION_KEY — "
            "либо ключ не тот, либо файл повреждён."
        ) from e


if __name__ == "__main__":
    new_key = Fernet.generate_key().decode("utf-8")
    print("Новый ключ шифрования сгенерирован. Добавьте эту строку в .env:\n")
    print(f"ENCRYPTION_KEY={new_key}\n")
    print("Сохраните ключ отдельно (например, в менеджере паролей) — без него "
          "расшифровать candidates.csv будет невозможно.")
