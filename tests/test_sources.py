"""Источники ответа: список источников, написанный моделью, отрезается в любом оформлении; номера — подряд."""
from __future__ import annotations

import pytest

from curator.sources import (ContextItem, SourceRef, citation_numbers, cite_numbers, process_citations, renumber,
                             strip_model_sources)

BODY = "loc выбирает строки по меткам, а iloc — по позициям [1]. Пример есть в лекции L2.2 [2].\n\n"


@pytest.mark.parametrize("tail", [
    "Источники\n1. Лекция L2.2",
    "**Источники:** [1] Лекция L2.2, [2] FAQ",
    "Источники: [1] Лекция L2.2",
    "Источник: [1] Лекция L2.2",
    "*Источники:*\n- [1] Лекция L2.2",
    "### Источники\n- [1] Лекция L2.2",
    "__Sources:__ [1] lecture",
    "**Источники**\n- [1] Лекция L2.2",
])
def test_model_sources_list_is_cut_in_any_form(tail: str):
    assert strip_model_sources(BODY + tail) == BODY.rstrip()


@pytest.mark.parametrize("tail", [
    "Источник данных: Kaggle, датасет о продажах кофейни.",  # обычное предложение, а не список источников
    "Источники данных бывают разными: CSV, Excel, базы данных.",
])
def test_sentence_starting_with_source_word_is_kept(tail: str):
    text = BODY + tail
    assert strip_model_sources(text) == text


def _item(n: int) -> ContextItem:
    return ContextItem(n, SourceRef("lecture", f"L{n}", f"Лекция {n}"), f"текст {n}")


def test_renumber_by_first_mention():
    text, items = renumber("Сначала [4], потом [6], снова [4][6]. Код `a[6]` не трогаем.", [_item(4), _item(6)])
    assert text == "Сначала [1], потом [2], снова [1][2]. Код `a[6]` не трогаем."
    assert [(i.n, i.source.ref_id) for i in items] == [(1, "L4"), (2, "L6")]


def test_renumber_keeps_text_when_already_consecutive():
    text = "Ответ [1] и [2] ."  # пробел перед точкой не трогаем, если номера не меняются
    assert renumber(text, [_item(1), _item(2)]) == (text, [_item(1), _item(2)])


def test_renumber_uncited_items_follow_in_old_order():
    text, items = renumber("Только [3].", [_item(1), _item(3), _item(5)])
    assert text == "Только [1]." and [i.source.ref_id for i in items] == ["L3", "L1", "L5"]
    assert [i.n for i in items] == [1, 2, 3]


# --- Ссылки-диапазоны и перечисления: «[1–5]», «[1-3]», «[1, 2]».

@pytest.mark.parametrize("inner, numbers", [
    ("1-3", [1, 2, 3]), ("1–5", [1, 2, 3, 4, 5]), ("2 — 3", [2, 3]), ("1, 2", [1, 2]), ("2; 4-5", [2, 4, 5]),
    ("7", [7]), ("0-10", None), ("5-1", None), ("0, 1, 2", None), ("0", None), ("1, 0", None),
])
def test_cite_numbers(inner: str, numbers: list[int] | None):
    assert cite_numbers(inner) == numbers


# --- Списки Python в тексте ответа вне `кода` — не ссылки (замечание ревью: «[10, 20, 30]» вырезался).

@pytest.mark.parametrize("text", [
    "sorted превращает список [30, 10, 20] в [10, 20, 30] [1].",
    "Индексы у списка из трёх элементов — это [0, 1, 2] [1].",
    "Срез вернёт [4, 5], если в списке 5 элементов [1].",
])
def test_list_literals_in_prose_are_kept(text: str):
    assert process_citations(text, 3)[0] == text
    assert renumber(text, [_item(1), _item(2), _item(3)])[0] == text
    assert citation_numbers(text, 3) == [1]


def test_single_unknown_number_still_removed_next_to_list():
    text, cited = process_citations("Список [10, 20] и ссылка [9], а правило — в лекции [1].", 3)
    assert text == "Список [10, 20] и ссылка, а правило — в лекции [1]." and cited == [1]


def test_blind_mode_removes_lists_too():
    """Слепым пакетам судей нужно убрать всё, что похоже на ссылку (keep_lists=False)."""
    assert process_citations("Правило [1, 2] и список [10, 20].", 0, keep_lists=False)[0] == "Правило и список."


def test_ranges_expanded_and_missing_numbers_dropped():
    text, cited = process_citations("Правило [2–4]. Ещё [1, 6] и [3-3].", 3)
    assert text == "Правило [2][3]. Ещё [1] и [3]."
    assert cited == [2, 3, 1]
    assert citation_numbers("См. [1–3] и [5]", 4) == [1, 2, 3]


def test_score_scale_in_brackets_is_not_a_citation():
    text = "Баллы ставят по шкале [0-10], а не по [5-1]."
    assert process_citations(text, 3) == (text, [])


def test_renumber_expands_ranges():
    text, items = renumber("Сначала [4-6], потом [4].", [_item(4), _item(5), _item(6)])
    assert text == "Сначала [1][2][3], потом [1]."
    assert [i.source.ref_id for i in items] == ["L4", "L5", "L6"]


# --- R3-2: пробел перед знаком убирается только на месте вырезанной ссылки.

@pytest.mark.parametrize("text", [
    "Сдайте ноутбук файлом .ipynb в LMS [1].",
    "Метод .copy() делает копию [2], а расширение .csv — у файла данных.",
    "Значения a , b и c ; d остаются как написаны [1] .",
])
def test_spaces_before_punctuation_kept_without_removed_citation(text: str):
    assert process_citations(text, 2)[0] == text
    shifted = text.replace("[1]", "[3]").replace("[2]", "[3]")  # перенумерация [3] → [1] переписывает текст
    assert renumber(shifted, [_item(3)])[0] == text.replace("[2]", "[1]")


@pytest.mark.parametrize("text, expected", [
    ("Сдайте файл .ipynb [9].", "Сдайте файл .ipynb."),
    ("Разделы L2.1 [9] и L2.2 [1] — там подробнее.", "Разделы L2.1 и L2.2 [1] — там подробнее."),
    ("Записи хранятся полгода [1], [7], [8].", "Записи хранятся полгода [1]."),
    ("Код `arr[1]` [9] не трогаем.", "Код `arr[1]` не трогаем."),
    ("[9] Строка со ссылкой в начале.\nВторая строка [9]\nТретья.",
     "Строка со ссылкой в начале.\nВторая строка\nТретья."),
    ("(подробнее в лекции [9])", "(подробнее в лекции)"),
])
def test_removed_citation_leaves_no_gap(text: str, expected: str):
    assert process_citations(text, 2)[0] == expected


def test_renumber_removes_unknown_number_without_gap():
    text, _ = renumber("Сначала [4], потом L2.1 [9] и [6].", [_item(4), _item(6)])
    assert text == "Сначала [1], потом L2.1 и [2]."
