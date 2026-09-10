"""
Посчитать, сколько токенов израсходовано на GigaChat.

    python -m scripts.token_report

Читает logs/logs.csv. Тексты вопросов и ответов там зашифрованы, но цифры
расхода лежат открыто, поэтому ключ шифрования для отчёта не нужен.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import paths  # noqa: E402


def _as_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def main():
    log_path = os.path.join(paths.LOGS_DIR, "logs.csv")
    if not os.path.exists(log_path):
        print("Лог ещё не создан — бот не обработал ни одного сообщения.")
        return

    with open(log_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    if not rows:
        print("Лог пуст.")
        return

    totals, cached, inputs, outputs = [], [], [], []
    for row in rows:
        total = _as_int(row.get("Всего токенов"))
        if total is None:
            continue
        totals.append(total)
        inputs.append(_as_int(row.get("Токены на вход")) or 0)
        outputs.append(_as_int(row.get("Токены на выход")) or 0)
        cached.append(_as_int(row.get("Кешировано токенов")) or 0)

    free = len(rows) - len(totals)

    print(f"Всего обращений к боту:      {len(rows)}")
    print(f"  из них дошло до GigaChat:  {len(totals)}")
    print(f"  обошлось без GigaChat:     {free}  (готовый ответ или шаг анкеты)")
    print()
    if not totals:
        print("К GigaChat пока не обращались — расход нулевой.")
        return

    print(f"Токенов на вход:             {sum(inputs)}")
    print(f"Токенов на выход:            {sum(outputs)}")
    print(f"Всего токенов:               {sum(totals)}")
    print(f"  из них из кеша (бесплатно): {sum(cached)}")
    print()
    print(f"В среднем на один вызов:     {sum(totals) // len(totals)}")
    if free:
        print(f"Доля бесплатных ответов:     {100 * free // len(rows)}%")


if __name__ == "__main__":
    main()
