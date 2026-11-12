"""eval/blind.py и eval/unblind.py: слепые пакеты (свой порядок у каждого судьи, ключ отдельно) и сводка оценок.
Без сети."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval import unblind
from eval.blind import (CRITERIA, build_packets, judge_seed, layouts, parse_judges, strip_sources, write_blind)
from eval.dataset import EvalItem, parse_items, write_json

CURATOR_TEXT = "Перенести срок я не могу [1]. Код `arr[1]` не трогаем.\n\n**Источники**\n\n- [1] FAQ: «Можно ли перенести»"


def _items(n: int) -> list[EvalItem]:
    intents = ("deadline", "grading_request", "no_answer", "material")
    return parse_items([{"id": f"q{k:02d}", "question": f"Вопрос {k}", "intent": intents[k % len(intents)],
                         "expected_points": ["пункт 1", "пункт 2"], "forbidden": ["выдумка"]} for k in range(n)])


def _answers(items: list[EvalItem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append({"id": item.id, "system": "curator", "answer": f"{CURATOR_TEXT} ({item.id})"})
        rows.append({"id": item.id, "system": "baseline", "answer": f"Ответ без источников ({item.id})."})
    return rows


def test_layouts_balanced_deterministic_and_different_per_judge():
    ids = [f"q{k}" for k in range(10)]
    judges = parse_judges("j1,j2,j3")
    first = layouts(ids, judges)
    assert [(lay.mapping, lay.order) for lay in first] == [(lay.mapping, lay.order) for lay in layouts(ids, judges)]
    maps = [[lay.mapping[q]["X"] for q in ids] for lay in first]
    assert len({tuple(m) for m in maps}) == 3  # у каждого судьи своя раскладка
    for m in maps:
        assert m.count("curator") == 5  # поровну под X
    assert len({tuple(lay.order) for lay in first}) == 3  # и свой порядок вопросов
    for lay in first:
        assert all(sorted(lay.mapping[q].values()) == ["baseline", "curator"] for q in ids)


def test_same_seed_for_two_judges_is_reshuffled():
    ids = [f"q{k}" for k in range(6)]
    a, b = layouts(ids, [("a", 7), ("b", 7)])
    assert [a.mapping[q] for q in ids] != [b.mapping[q] for q in ids] and b.reshuffles >= 1


def test_parse_judges():
    assert parse_judges("j1:101, j2:202") == [("j1", 101), ("j2", 202)]
    assert parse_judges("j1") == [("j1", judge_seed("j1"))] and judge_seed("j1") != judge_seed("j2")
    with pytest.raises(ValueError):
        parse_judges("j1,j1")
    with pytest.raises(ValueError):
        parse_judges("плохое имя!")


def test_strip_sources_removes_block_and_citations_but_not_code():
    assert strip_sources(CURATOR_TEXT) == "Перенести срок я не могу. Код `arr[1]` не трогаем."


def test_strip_sources_removes_citation_ranges_and_lists():
    """Диапазоны и перечисления ссылок тоже выдают куратора судье — вырезаются, как и одиночные."""
    text = "В материалах этого нет [1–5]. Правило [1-3], ещё [1, 2] и [2; 4].\n\n**Источники**\n\n- [1] FAQ курса"
    stripped = strip_sources(text)
    assert stripped.startswith("В материалах этого нет. Правило, ещё")
    assert "[" not in stripped and "Источники" not in stripped


def test_packets_are_blind_and_key_is_separate(tmp_path: Path):
    items = _items(5)
    answers = _answers(items)
    answers[-1]["error"] = "ConnectionError"  # у q04 нет ответа наивного RAG — вопрос пропускается
    packets, key = build_packets(items, answers, parse_judges("j1,j2"))
    assert key["skipped"] == {"q04": "нет ответа: baseline"}
    for name, packet in packets.items():
        dumped = json.dumps(packet, ensure_ascii=False).lower()
        assert "curator" not in dumped and "baseline" not in dumped  # имён систем в пакете нет
        texts = [a for i in packet["items"] for a in i["answers"].values()]
        assert not any("**Источники**" in t or "[1]" in t.replace("`arr[1]`", "") for t in texts)
        assert [i["id"] for i in packet["items"]] == key["judges"][name]["order"]
        assert len(packet["items"]) == 4 and packet["instructions"] and packet["response_format"]["packet"] == name
        for item in packet["items"]:
            mapping = key["judges"][name]["map"][item["id"]]
            curator_name = "X" if mapping["X"] == "curator" else "Y"
            assert item["answers"][curator_name].startswith("Перенести срок я не могу.")
    blind = write_blind(tmp_path, packets, key)
    assert sorted(p.name for p in (blind / "packets").iterdir()) == ["j1.json", "j1.md", "j2.json", "j2.md"]
    assert (blind / "key.json").is_file() and (blind / "scores").is_dir()
    assert "Ответ X" in (blind / "packets" / "j1.md").read_text(encoding="utf-8")


def test_keep_sources_leaves_answers_as_is():
    items = _items(2)
    packets, key = build_packets(items, _answers(items), parse_judges("j1"), keep_sources=True)
    assert key["sources_stripped"] is False
    assert any("**Источники**" in a for i in packets["j1"]["items"] for a in i["answers"].values())


def _scores(prefer: str, mapping: dict[str, str]) -> dict[str, Any]:
    good = {c: 5 for c in CRITERIA} | {"points": [1, 2], "forbidden": []}
    bad = {c: 2 for c in CRITERIA} | {"points": [1], "forbidden": [1]}
    winner = "X" if mapping["X"] == prefer else "Y"
    other = "Y" if winner == "X" else "X"
    return {winner: good, other: bad, "winner": winner}


def test_unblind_round_trip(tmp_path: Path):
    items = _items(8)
    set_path = tmp_path / "set.json"
    write_json(set_path, [{"id": i.id, "question": i.question, "intent": i.intent,
                           "expected_points": list(i.expected_points), "forbidden": list(i.forbidden)}
                          for i in items])
    packets, key = build_packets(items, _answers(items), parse_judges("j1,j2"))
    blind = write_blind(tmp_path, packets, key)
    for name in ("j1", "j2"):
        ratings = [{"id": qid, **_scores("curator", key["judges"][name]["map"][qid])}
                   for qid in key["judges"][name]["order"]]
        if name == "j2":
            ratings[0]["X"]["correctness"] = 7  # неверная оценка — не считается
            ratings.pop()  # и одного вопроса нет
        write_json(blind / "scores" / f"{name}.json", {"packet": name, "ratings": ratings})

    assert unblind.main([str(tmp_path), "--set", str(set_path)]) == 0
    summary = json.loads((blind / "unblind.json").read_text(encoding="utf-8"))
    assert summary["ratings"] == 8 + 6
    cur, base = summary["systems"]["curator"], summary["systems"]["baseline"]
    assert cur["wins"] == 14 and base["wins"] == 0 and summary["ties"] == 0
    assert cur["overall"] == 5.0 and base["overall"] == 2.0
    assert cur["points_share"] == 1.0 and base["points_share"] == 0.5 and base["forbidden_share"] == 1.0
    j1 = summary["by_judge"]["j1"]
    assert j1["x_wins"] == 4 and j1["y_wins"] == 4  # победы X и Y поровну: судья выбирал систему, а не позицию
    assert summary["agreement"]["unanimous"] == summary["agreement"]["questions_multi"] == 6
    assert any("correctness = 7" in p for p in summary["problems"])
    assert any("нет оценок вопросов" in p for p in summary["problems"])
    md = (blind / "unblind.md").read_text(encoding="utf-8")
    assert "| Показатель | curator | baseline |" in md and "## По сценариям" in md and "## Что не засчитано" in md


def test_unblind_rejects_unknown_judge_and_bad_winner(tmp_path: Path):
    items = _items(2)
    _, key = build_packets(items, _answers(items), parse_judges("j1"))
    bad = tmp_path / "j9.json"
    write_json(bad, {"ratings": []})
    wrong = tmp_path / "j1.json"
    qid = key["judges"]["j1"]["order"][0]
    record = {"id": qid, **_scores("curator", key["judges"]["j1"]["map"][qid])}
    record["winner"] = "curator"  # имя системы вместо X/Y
    write_json(wrong, {"packet": "j1", "ratings": [record]})
    collected = unblind.collect([bad, wrong], key, items)
    assert collected.ratings == []
    assert any("судьи 'j9' нет в ключе" in p for p in collected.problems)
    assert any("winner 'curator'" in p for p in collected.problems)


def test_strip_sources_leaves_no_double_spaces_or_stray_commas():
    """След вырезанных ссылок («L2.1  и», «доступна,,.») выдавал бы судье куратора; пробелы в коде и в остальном
    тексте не меняются."""
    text = ("Повторите разделы L2.1 [1] и L2.2 [2] — там подробно. Запись уже в LMS [1], [2], [3].\n"
            "Сдайте файл .ipynb [2].\n\n```python\nx = df.iloc[0]  # первая строка\n```\n\n**Источники**\n\n- [1] FAQ")
    stripped = strip_sources(text)
    assert stripped == ("Повторите разделы L2.1 и L2.2 — там подробно. Запись уже в LMS.\nСдайте файл .ipynb.\n\n"
                        "```python\nx = df.iloc[0]  # первая строка\n```")
