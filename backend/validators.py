"""
Валидация значений полей анкеты перед сохранением в candidates.csv.

Что проверяется: формат (телефон, даты, паспорт, СНИЛС, ИНН, числовые поля) —
достаточно, чтобы отсечь явный мусор/опечатки. Контрольная сумма СНИЛС/ИНН
намеренно НЕ проверяется: неверно реализованная проверка контрольной суммы
может ошибочно отклонить настоящий валидный номер, а это хуже, чем пропустить
редкую опечатку. Если понадобится проверка контрольной суммы — её нужно
сверять с официальным алгоритмом ФНС/ПФР, а не реализовывать по памяти.

Свободнотекстовые поля (адрес, образование и т.п.) проверяются только на
непустоту и разумную максимальную длину — жёсткий формат для них не задан
и не нужен.
"""
import re
from datetime import datetime


class FieldValidationError(ValueError):
    def __init__(self, field_name: str, message: str):
        self.field_name = field_name
        self.message = message
        super().__init__(f"{field_name}: {message}")


_MAX_TEXT_LEN = 500
_MAX_SHORT_LEN = 120


def _require_nonempty(value: str, field_name: str, max_len: int = _MAX_TEXT_LEN) -> str:
    v = value.strip()
    if not v:
        raise FieldValidationError(field_name, "значение не может быть пустым.")
    if len(v) > max_len:
        raise FieldValidationError(field_name, f"слишком длинное значение (максимум {max_len} символов).")
    return v


# Максимальная длина для ФИО-подобных полей. 40 символов с запасом хватает
# на самые длинные реальные русские отчества/фамилии (например
# "Александрович" — 13 букв), но уже отсекает совсем неправдоподобные
# значения (мягкое, не жёсткое ограничение — см. заголовок файла).
_MAX_NAME_LEN = 40


def _validate_name_part(value: str, field_name: str) -> str:
    v = _require_nonempty(value, field_name, _MAX_NAME_LEN)
    if not re.fullmatch(r"[А-ЯЁа-яёA-Za-z\-\s']+", v):
        raise FieldValidationError(field_name, "допустимы только буквы, дефис и пробел.")
    return v


def _validate_date(value: str, field_name: str, *, allow_future: bool = False,
                    min_year: int = 1900) -> str:
    v = _require_nonempty(value, field_name, 32)
    parsed = None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            parsed = datetime.strptime(v, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        raise FieldValidationError(field_name, "не удалось распознать дату (ожидается формат ДД.ММ.ГГГГ).")
    if parsed.year < min_year:
        raise FieldValidationError(field_name, f"дата раньше {min_year} года — похоже на опечатку.")
    if not allow_future and parsed > datetime.now():
        raise FieldValidationError(field_name, "дата не может быть в будущем.")
    return v


def _validate_phone(value: str, field_name: str) -> str:
    v = _require_nonempty(value, field_name, 32)
    digits = re.sub(r"\D", "", v)
    if len(digits) == 11 and digits[0] in ("7", "8"):
        pass
    elif len(digits) == 10:
        pass
    else:
        raise FieldValidationError(field_name, "похоже на неверный номер телефона (ожидается 10-11 цифр).")
    return v


def _validate_telegram(value: str, field_name: str) -> str:
    v = _require_nonempty(value, field_name, _MAX_SHORT_LEN)
    if not re.fullmatch(r"@?[A-Za-z0-9_]{5,32}", v):
        raise FieldValidationError(
            field_name, "похоже на неверный Telegram-юзернейм (латиница/цифры/_, 5-32 символа)."
        )
    return v


def _validate_digits(value: str, field_name: str, length: int, label: str) -> str:
    v = _require_nonempty(value, field_name, 32)
    digits = re.sub(r"\D", "", v)
    if len(digits) != length:
        raise FieldValidationError(field_name, f"{label} должен содержать {length} цифр, получено {len(digits)}.")
    return v


def _validate_dept_code(value: str, field_name: str) -> str:
    v = _require_nonempty(value, field_name, 16)
    digits = re.sub(r"\D", "", v)
    if len(digits) != 6:
        raise FieldValidationError(field_name, "код подразделения должен содержать 6 цифр (формат XXX-XXX).")
    return v


def _validate_number_range(value: str, field_name: str, lo: float, hi: float, label: str) -> str:
    v = _require_nonempty(value, field_name, 16)
    digits = re.sub(r"[^\d.,]", "", v).replace(",", ".")
    try:
        num = float(digits)
    except ValueError:
        raise FieldValidationError(field_name, f"{label} должен быть числом.")
    if not (lo <= num <= hi):
        raise FieldValidationError(field_name, f"{label} вне разумного диапазона ({lo:.0f}-{hi:.0f}).")
    return v


def _validate_clothing_size(value: str, field_name: str) -> str:
    v = _require_nonempty(value, field_name, 16)
    if re.fullmatch(r"\d{2}(-\d{2})?", v):
        return v
    if v.upper() in ("XS", "S", "M", "L", "XL", "XXL", "XXXL"):
        return v
    raise FieldValidationError(field_name, "укажите размер числом (например 50) или буквенно (например L).")


# --- Реестр валидаторов по имени поля (см. candidates.FIELD_ORDER) --------

_VALIDATORS = {
    "Фамилия": _validate_name_part,
    "Имя": _validate_name_part,
    "Отчество": _validate_name_part,
    "Дата рождения": lambda v, f: _validate_date(v, f, allow_future=False),
    "Телефон": _validate_phone,
    "Telegram": _validate_telegram,
    "Паспорт серия": lambda v, f: _validate_digits(v, f, 4, "серия паспорта"),
    "Паспорт номер": lambda v, f: _validate_digits(v, f, 6, "номер паспорта"),
    "Дата выдачи": lambda v, f: _validate_date(v, f, allow_future=False),
    "Код подразделения выдачи паспорта": _validate_dept_code,
    "СНИЛС": lambda v, f: _validate_digits(v, f, 11, "СНИЛС"),
    "ИНН": lambda v, f: _validate_digits(v, f, 12, "ИНН"),
    "Рост": lambda v, f: _validate_number_range(v, f, 100, 230, "рост (см)"),
    "Вес": lambda v, f: _validate_number_range(v, f, 30, 250, "вес (кг)"),
    "Размер одежды": _validate_clothing_size,
    "Размер обуви": lambda v, f: _validate_number_range(v, f, 30, 50, "размер обуви"),
}


def validate_field(field_name: str, value: str) -> str:
    """Проверяет значение поля. Возвращает очищенное значение или бросает
    FieldValidationError с понятным (по-русски) сообщением. Поля без
    специального валидатора (свободный текст: адреса, опыт работы,
    образование, судимость, ограничения по здоровью и т.д.) проверяются
    только на непустоту и разумную длину."""
    validator = _VALIDATORS.get(field_name)
    if validator:
        return validator(value, field_name)
    return _require_nonempty(value, field_name)
