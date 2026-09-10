"""
Расшифровать анкеты кандидатов и выгрузить их в обычный CSV для Excel.

    python -m scripts.export_candidates

Создаёт файл candidates/candidates_readable.csv — тот же набор данных, но
БЕЗ шифрования, чтобы его можно было открыть Excel'ем.

ВАЖНО: этот файл содержит паспорта, СНИЛС и ИНН в открытом виде. Он
намеренно не создаётся автоматически и не попадает в git (папка candidates/
целиком в .gitignore). После работы с ним файл лучше удалить — исходные
данные никуда не денутся, они остаются в зашифрованном candidates.csv.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import candidates, paths  # noqa: E402


def main():
    fields, table = candidates._load_table()
    if not table:
        print("В таблице нет ни одного кандидата.")
        return

    out_path = os.path.join(paths.CANDIDATS_DIR, "candidates_readable.csv")
    ids = sorted(table.keys(), key=lambda x: int(x) if x.isdigit() else x)

    # utf-8-sig, а не utf-8: без BOM Excel на Windows открывает файл в
    # однобайтовой кодировке и показывает кириллицу кракозябрами.
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Поле"] + [f"Кандидат {i}" for i in ids])
        for field in fields:
            writer.writerow([field] + [table[i].get(field, "") for i in ids])

    print(f"Готово: {out_path}")
    print(f"Кандидатов выгружено: {len(ids)}")
    print()
    print("Файл НЕ зашифрован и содержит персональные данные.")
    print("Удалите его, когда закончите работу.")


if __name__ == "__main__":
    main()
