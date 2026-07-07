import csv
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.csv")

# Поля точно соответствуют карточке кандидата (карточка_кандидата.xlsx)
FIELDNAMES = [
    "Источник",
    "ДАТА заполнения",
    "Фамилия",
    "Имя",
    "Отчество",
    "Дата рождения",
    "Телефон",
    "Telegram",
    "Паспорт серия",
    "Паспорт номер",
    "Кем выдан",
    "Дата выдачи",
    "Код подразделения выдачи паспорта",
    "Адрес регистрации",
    "Адрес проживания",
    "СНИЛС",
    "ИНН",
    "Гражданство",
    "Город проживания",
    "Желаемая должность",
    "Опыт работы",
    "Последнее место работы",
    "Образование/удостоверения/напиши что есть",
    "Рост",
    "Вес",
    "Размер одежды",
    "Размер обуви",
    "Судимость",
    "Административный надзор",
    "Ограничения по здоровью",
    "Готовность к вахте, когда готов",
]


def save_candidate(source: str, data: dict):
    """
    source — откуда пришла анкета: "site" или "telegram".
    data — словарь с любыми полями из FIELDNAMES (кроме "Источник" и "ДАТА заполнения" —
    они проставляются автоматически). Отсутствующие поля останутся пустыми.
    """
    file_exists = os.path.exists(CANDIDATES_PATH)

    row = {field: "" for field in FIELDNAMES}
    row["Источник"] = source
    row["ДАТА заполнения"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for key, value in data.items():
        if key in row:
            row[key] = value

    with open(CANDIDATES_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_all_candidates():
    if not os.path.exists(CANDIDATES_PATH):
        return []

    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)