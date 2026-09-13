"""Проверка формата ответов кандидата."""
import pytest

from backend.validators import FieldValidationError, validate_field


@pytest.mark.parametrize("raw,expected", [
    ("89991234567", "+79991234567"),
    ("+7 (999) 123-45-67", "+79991234567"),
    ("8-999-123-45-67", "+79991234567"),
    ("9991234567", "+79991234567"),
    ("+375 29 123-45-67", "+375291234567"),   # Беларусь
])
def test_telefon_privoditsya_k_odnomu_vidu(raw, expected):
    """Без нормализации HR не находит кандидата поиском по таблице."""
    assert validate_field("Телефон", raw) == expected


def test_telefon_musor_otklonyaetsya():
    with pytest.raises(FieldValidationError):
        validate_field("Телефон", "12345")


@pytest.mark.parametrize("field,raw", [
    ("Судимость", "Есть"),
    ("Судимость", "да"),
    ("Ограничения по здоровью", "Да есть"),
    ("Образование/удостоверения/напиши что есть", "есть"),
])
def test_odnoslozhnyy_otvet_trebuet_podrobnostey(field, raw):
    """«Есть» без подробностей бесполезно для HR — нужен переспрос."""
    with pytest.raises(FieldValidationError):
        validate_field(field, raw)


@pytest.mark.parametrize("field,raw", [
    ("Судимость", "нет"),
    ("Судимость", "ст. 158, срок закончился в 2019"),
    ("Ограничения по здоровью", "грыжа поясничного отдела"),
])
def test_razvernutyy_otvet_prinimaetsya(field, raw):
    assert validate_field(field, raw) == raw


def test_inn_mozhno_net():
    """У граждан Беларуси ИНН может не быть — это не ошибка ввода."""
    assert validate_field("ИНН", "нет") == "нет"
    assert validate_field("ИНН", "отсутствует") == "нет"
    assert validate_field("ИНН", "123456789012") == "123456789012"


@pytest.mark.parametrize("field,raw", [
    ("Размер обуви", "42-43"),
    ("Рост", "180 см"),
    ("Вес", "75 кг"),
])
def test_diapazony_i_edinicy_prinimayutsya(field, raw):
    """Люди пишут диапазоны и единицы — склеивать цифры нельзя."""
    assert validate_field(field, raw) == raw
