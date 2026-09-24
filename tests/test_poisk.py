"""Контрольный набор: попадает ли вопрос в НУЖНЫЙ готовый ответ.

Зачем это отдельно от остальных тестов. Готовый ответ из faqs.json уходит
кандидату НАПРЯМУЮ, без нейросети, по первому результату поиска. Если
«сколько платят изолировщику» поймает ответ про монтажника, человек увидит
уверенный неверный ответ — и исправить его будет нечем, модель в этот путь
даже не заглядывает.

А в faqs.json есть тройки почти одинаковых вопросов, различающихся одним
словом: зарплата монтажника, альпиниста, изолировщика; вахта монтажника и
альпиниста; что делает каждый из троих. Вектор строится из вопроса,
ключевых слов и ответа — а названия профессий там малая часть текста.

ВАЖНО: тесту нужна настоящая модель поиска, она скачивается с huggingface
и весит около 470 МБ. Если её нет, тест пропускается — на машине, где
проект работает, она уже в кеше и тест выполнится по-настоящему.
"""
import pytest


# Вопрос кандидата -> слово, которое ОБЯЗАНО быть в найденном ответе.
CONTROL_SET = [
    ("сколько платят монтажнику",              "монтаж"),
    ("сколько зарабатывает монтажник лесов",   "монтаж"),
    ("зарплата у промышленного альпиниста",    "альпинист"),
    ("сколько получает альпинист",             "альпинист"),
    ("сколько платят изолировщику",            "изолиров"),
    ("зарплата изолировщика трубопровода",     "изолиров"),
    ("какая вахта у монтажника",               "монтаж"),
    ("какая вахта у альпиниста",               "альпинист"),
    ("что делает монтажник",                   "монтаж"),
    ("чем занимается промышленный альпинист",  "альпинист"),
    ("что делает изолировщик",                 "изолиров"),
]


@pytest.fixture(scope="module")
def faq_search():
    """Настоящий поиск по faqs.json — с реальной моделью и индексом."""
    pytest.importorskip("faiss")

    # Обязательно НАСТОЯЩАЯ модель, а не заглушка из conftest.
    #
    # Общая обвязка подменяет sentence_transformers пустышкой, чтобы
    # остальные тесты не тянули 470 МБ. Если взять её здесь, тест будет
    # сравнивать случайные числа и «проходить» или «падать» бессмысленно.
    # У настоящего пакета есть файл на диске, у заглушки — нет.
    import importlib
    import os
    import sys

    module = sys.modules.get("sentence_transformers")
    if module is not None and not getattr(module, "__file__", None):
        del sys.modules["sentence_transformers"]
        for name in [m for m in list(sys.modules) if m.startswith("backend")]:
            del sys.modules[name]
    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
    except ImportError:
        pytest.skip("sentence-transformers не установлен")
    if not getattr(sentence_transformers, "__file__", None):
        pytest.skip("вместо модели поиска стоит заглушка")

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from backend import build_index
        from backend.rag_index import load_index, search_similar
        build_index.main(force=False)
        index, meta = load_index(build_index.INDEX_PATH, build_index.META_PATH)
        model = sentence_transformers.SentenceTransformer(build_index.EMBEDDING_MODEL_NAME)
    except Exception as e:
        pytest.skip(f"модель поиска недоступна: {e}")

    def search(question, k=1):
        """Возвращает (список результатов, близость лучшего).

        Близость считается здесь, а не берётся из search_similar: та
        возвращает только сами записи, без числа. Лезть в рабочий код ради
        теста не стоит — проще посчитать на месте."""
        vector = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        results = search_similar(index, meta, vector, k=k, min_similarity=0.0)
        scores, _ = index.search(vector, k)
        best = float(scores[0][0]) if len(scores[0]) else None
        return results, best

    return search


@pytest.mark.parametrize("question,must_contain", CONTROL_SET)
def test_vopros_nahodit_svoyu_professiyu(faq_search, question, must_contain):
    """Первый результат поиска должен относиться к той профессии, о которой
    спросили. Это самое уязвимое место: ответ уходит кандидату как есть."""
    results, _ = faq_search(question, k=1)
    assert results, f"поиск вообще ничего не нашёл на «{question}»"

    found = (results[0]["question"] + " " + results[0]["answer"]).lower()
    assert must_contain in found, (
        f"на «{question}» найден ответ про другое:\n"
        f"   {results[0]['question']}\n"
        f"   ожидали упоминание «{must_contain}»"
    )


def test_porog_otsekaet_postoronnee(faq_search):
    """Вопрос не по теме не должен получать готовый ответ.

    Порог FAQ_MIN_SIMILARITY существует именно для этого: лучше отправить
    вопрос нейросети, чем уверенно ответить не по делу."""
    from backend.assistant import FAQ_MIN_SIMILARITY

    for question in ("какая погода в Мытищах", "рецепт борща", "курс доллара"):
        _, best = faq_search(question, k=1)
        assert best is not None, "поиск не вернул близость"
        assert best < FAQ_MIN_SIMILARITY, (
            f"«{question}» получил готовый ответ с близостью {best:.2f} "
            f"при пороге {FAQ_MIN_SIMILARITY} — а должен был уйти нейросети"
        )
