"""
Полное удаление данных кандидата по его ID — по запросу кандидата (право на
удаление персональных данных, 152-ФЗ).

Запуск из корня проекта (там же, где лежит .env):
    python -m scripts.delete_candidate <ID_кандидата>

ID кандидата — это номер колонки в candidates.csv (после расшифровки), не
телефон и не ФИО. Посмотреть ID можно, открыв анкету через обычный процесс
работы с ботом, либо явно из кода (candidates.get_card).

Стирает безвозвратно:
  - строку этого кандидата в HR/candidates/candidates.csv;
  - его запись(и) в HR/candidates/candidate_sessions.json (привязку сессии к ID);
  - файл(ы) истории переписки в папке conversations.

НЕ трогает logs/logs.csv — решение: логи хранить полностью, без изменений.
"""
import argparse
import sys

from backend import candidates


def main():
    parser = argparse.ArgumentParser(description="Полное удаление данных кандидата по ID.")
    parser.add_argument("candidate_id", help="ID кандидата — номер колонки в candidates.csv")
    parser.add_argument("--yes", action="store_true", help="не спрашивать подтверждение")
    args = parser.parse_args()

    card = candidates.get_card(args.candidate_id)
    if not card:
        print(f"Кандидат с ID {args.candidate_id!r} не найден.")
        sys.exit(1)

    fio = " ".join(
        part for part in (card.get("Фамилия"), card.get("Имя"), card.get("Отчество")) if part
    ) or "(ФИО ещё не заполнено)"
    print(f"Найден кандидат {args.candidate_id}: {fio}")

    if not args.yes:
        answer = input("Удалить ВСЕ данные этого кандидата безвозвратно? (да/нет): ").strip().lower()
        if answer not in ("да", "yes", "y", "д"):
            print("Отменено, ничего не удалено.")
            sys.exit(0)

    result = candidates.delete_candidate(args.candidate_id)
    print("Готово. Удалено:")
    print(f"  - строка в candidates.csv: {'да' if result['removed_from_table'] else 'уже отсутствовала'}")
    print(f"  - записи в candidate_sessions.json: {result['removed_sessions'] or 'не найдено'}")
    print(f"  - файлы истории переписки: {result['removed_conversations'] or 'не найдено'}")
    print("logs/logs.csv не менялся — по решению логи и историю хранить полностью.")


if __name__ == "__main__":
    main()
