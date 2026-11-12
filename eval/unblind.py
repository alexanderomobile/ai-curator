"""Раскрытие слепой оценки: оценки судей (JSON) + ключ blind/key.json → таблицы по системам, сценариям и судьям.

    python -m eval.unblind <прогон>                              # оценки из blind/scores/*.json
    python -m eval.unblind <прогон> --scores a.json b.json

Файл судьи — JSON по образцу response_format из пакета:
    {"packet": "j1", "ratings": [{"id": "d01", "X": {"correctness": 4, "completeness": 3, "safety": 5,
      "helpfulness": 4, "points": [1, 2], "forbidden": []}, "Y": {...}, "winner": "X", "comment": "..."}]}
Имя судьи — поле packet, а если его нет — имя файла. Неверные записи не считаются и перечисляются в отчёте.
Пишет blind/unblind.md и blind/unblind.json.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from eval.blind import CRITERIA, CRITERIA_TITLES, NAMES, PAIR
from eval.dataset import INTENTS, DatasetError, EvalItem, read_json, resolve_run_dir, run_items, write_json

WINNERS = (*NAMES, "tie")


@dataclass
class Rating:
    judge: str
    id: str
    intent: str
    scores: dict[str, dict[str, int]]  # система → критерий → 1..5
    points: dict[str, int]  # система → сколько ожидаемых пунктов раскрыто
    forbidden: dict[str, int]  # система → сколько запретов нарушено
    winner: str  # система или «tie»
    winner_name: str  # X | Y | tie — для проверки смещения позиции
    expected: int  # сколько ожидаемых пунктов у вопроса


@dataclass
class Collected:
    ratings: list[Rating] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _int_list(value: Any, upper: int, where: str, problems: list[str]) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, int) and not isinstance(v, bool)
                                              for v in cast(list[Any], value)):
        problems.append(f"{where}: ожидается список номеров")
        return []
    numbers = sorted(set(cast(list[int], value)))
    bad = [n for n in numbers if not 1 <= n <= upper]
    if bad:
        problems.append(f"{where}: номера вне 1..{upper}: {bad}")
    return [n for n in numbers if 1 <= n <= upper]


def parse_rating(raw: Any, judge: str, key_map: Mapping[str, Mapping[str, str]], items: Mapping[str, EvalItem],
                 problems: list[str]) -> Rating | None:
    if not isinstance(raw, dict):
        problems.append(f"{judge}: запись оценки — не объект")
        return None
    data = cast(dict[str, Any], raw)
    qid = str(data.get("id", ""))
    where = f"{judge}/{qid or '?'}"
    if qid not in key_map:
        problems.append(f"{where}: такого вопроса в пакете судьи нет")
        return None
    item = items.get(qid)
    if item is None:
        problems.append(f"{where}: вопроса нет в наборе")
        return None
    mapping = key_map[qid]
    winner_name = data.get("winner")
    if winner_name not in WINNERS:
        problems.append(f"{where}: winner {winner_name!r} — X, Y или tie")
        return None
    scores: dict[str, dict[str, int]] = {}
    points: dict[str, int] = {}
    forbidden: dict[str, int] = {}
    for name in NAMES:
        block = data.get(name)
        if not isinstance(block, dict):
            problems.append(f"{where}: нет оценок ответа {name}")
            return None
        b = cast(dict[str, Any], block)
        values: dict[str, int] = {}
        for criterion in CRITERIA:
            value = b.get(criterion)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                problems.append(f"{where}: {name}.{criterion} = {value!r} — целое 1-5")
                return None
            values[criterion] = value
        system = mapping[name]
        scores[system] = values
        points[system] = len(_int_list(b.get("points"), len(item.expected_points), f"{where}: {name}.points",
                                       problems))
        forbidden[system] = len(_int_list(b.get("forbidden"), len(item.forbidden), f"{where}: {name}.forbidden",
                                          problems))
    winner = mapping[str(winner_name)] if winner_name in NAMES else "tie"
    return Rating(judge, qid, item.intent, scores, points, forbidden, winner, str(winner_name),
                  len(item.expected_points))


def collect(files: Iterable[Path], key: Mapping[str, Any], items: Sequence[EvalItem]) -> Collected:
    out = Collected()
    by_id = {i.id: i for i in items}
    judges = cast(dict[str, dict[str, Any]], key.get("judges") or {})
    for path in files:
        try:
            data = read_json(path)
        except (OSError, ValueError) as exc:
            out.problems.append(f"{path.name}: не читается как JSON ({exc})")
            continue
        doc = cast(dict[str, Any], data) if isinstance(data, dict) else {"ratings": data}
        judge = str(doc.get("packet") or path.stem)
        if judge not in judges:
            out.problems.append(f"{path.name}: судьи {judge!r} нет в ключе (есть: {', '.join(judges)})")
            continue
        key_map = cast(dict[str, dict[str, str]], judges[judge].get("map") or {})
        ratings = doc.get("ratings")
        if not isinstance(ratings, list):
            out.problems.append(f"{path.name}: нет списка ratings")
            continue
        seen: set[str] = set()
        for raw in cast(list[Any], ratings):
            rating = parse_rating(raw, judge, key_map, by_id, out.problems)
            if rating is None:
                continue
            if rating.id in seen:
                out.problems.append(f"{judge}/{rating.id}: оценка повторяется — считается первая")
                continue
            seen.add(rating.id)
            out.ratings.append(rating)
        missing = [qid for qid in key_map if qid not in seen]
        if missing:
            out.problems.append(f"{judge}: нет оценок вопросов {', '.join(missing)}")
    return out


# --- сводка

def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def overall(scores: Mapping[str, int]) -> float:
    return sum(scores[c] for c in CRITERIA) / len(CRITERIA)


def system_stats(ratings: Sequence[Rating], system: str) -> dict[str, Any]:
    stats: dict[str, Any] = {c: _mean([r.scores[system][c] for r in ratings]) for c in CRITERIA}
    stats["overall"] = _mean([overall(r.scores[system]) for r in ratings])
    expected = sum(r.expected for r in ratings)
    stats["points_share"] = round(sum(r.points[system] for r in ratings) / expected, 3) if expected else None
    stats["forbidden_share"] = round(sum(1 for r in ratings if r.forbidden[system]) / len(ratings), 3) \
        if ratings else None
    stats["wins"] = sum(1 for r in ratings if r.winner == system)
    return stats


def summarize(collected: Collected) -> dict[str, Any]:
    ratings = collected.ratings
    out: dict[str, Any] = {"ratings": len(ratings), "judges": sorted({r.judge for r in ratings}),
                           "questions": len({r.id for r in ratings}), "ties": sum(1 for r in ratings if r.winner == "tie"),
                           "systems": {s: system_stats(ratings, s) for s in PAIR}, "by_intent": {}, "by_judge": {},
                           "problems": list(collected.problems)}
    for intent in INTENTS:
        scoped = [r for r in ratings if r.intent == intent]
        if scoped:
            out["by_intent"][intent] = {"questions": len({r.id for r in scoped}), "ratings": len(scoped),
                                        "ties": sum(1 for r in scoped if r.winner == "tie"),
                                        **{s: system_stats(scoped, s) for s in PAIR}}
    for judge in out["judges"]:
        mine = [r for r in ratings if r.judge == judge]
        out["by_judge"][judge] = {"ratings": len(mine), "ties": sum(1 for r in mine if r.winner == "tie"),
                                  "x_wins": sum(1 for r in mine if r.winner_name == "X"),
                                  "y_wins": sum(1 for r in mine if r.winner_name == "Y"),
                                  **{s: {"wins": sum(1 for r in mine if r.winner == s),
                                         "overall": _mean([overall(r.scores[s]) for r in mine])} for s in PAIR}}
    by_question: dict[str, list[str]] = {}
    for r in ratings:
        by_question.setdefault(r.id, []).append(r.winner)
    multi = {qid: w for qid, w in by_question.items() if len(w) >= 2}
    majority: Counter[str] = Counter()
    for winners in by_question.values():
        top, count = Counter(winners).most_common(1)[0]
        majority[top if count * 2 > len(winners) else "no_majority"] += 1  # больше половины судей
    out["agreement"] = {"questions_multi": len(multi),
                        "unanimous": sum(1 for w in multi.values() if len(set(w)) == 1),
                        "majority": dict(majority)}
    return out


def _pct(part: int, whole: int) -> str:
    return f"{part} ({part / whole * 100:.0f}%)" if whole else "—"


def _num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _share(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(r) + " |" for r in rows]


def render(summary: Mapping[str, Any], run: str) -> str:
    n = int(summary["ratings"])
    systems = cast(dict[str, dict[str, Any]], summary["systems"])
    lines = [f"# Слепая оценка судьями — прогон {run}", "",
             f"Судей: {len(summary['judges'])} ({', '.join(summary['judges']) or '—'}); вопросов: "
             f"{summary['questions']}; сравнений (судья × вопрос): {n}. Оценки 1-5, среднее по всем сравнениям.", ""]
    rows = [[CRITERIA_TITLES[c], *(_num(systems[s][c]) for s in PAIR)] for c in CRITERIA]
    rows += [["Среднее по четырём критериям", *(_num(systems[s]["overall"]) for s in PAIR)],
             ["Раскрыто ожидаемых пунктов", *(_share(systems[s]["points_share"]) for s in PAIR)],
             ["Ответов с нарушенным запретом", *(_share(systems[s]["forbidden_share"]) for s in PAIR)],
             ["Победы", *(_pct(int(systems[s]["wins"]), n) for s in PAIR)]]
    lines += ["## По системам", ""] + _table(["Показатель", *PAIR], rows)
    lines += ["", f"Ничьих: {_pct(int(summary['ties']), n)}.", "", "## По сценариям", ""]
    body: list[list[str]] = []
    for intent, data in cast(dict[str, dict[str, Any]], summary["by_intent"]).items():
        k = int(data["ratings"])
        body.append([intent, str(data["questions"]), *(_num(data[s]["overall"]) for s in PAIR),
                     *(_pct(int(data[s]["wins"]), k) for s in PAIR), _pct(int(data["ties"]), k)])
    lines += _table(["intent", "вопросов", *(f"{s}: среднее" for s in PAIR), *(f"{s}: побед" for s in PAIR),
                     "ничьих"], body)
    lines += ["", "## По судьям", "",
              "«X побед» — проверка смещения позиции: у каждого судьи под X поровну обеих систем, поэтому заметный "
              "перекос в пользу X или Y говорит о судье, а не о системах.", ""]
    body = []
    for judge, data in cast(dict[str, dict[str, Any]], summary["by_judge"]).items():
        k = int(data["ratings"])
        body.append([judge, str(k), *(_pct(int(data[s]["wins"]), k) for s in PAIR), _pct(int(data["ties"]), k),
                     *(_num(data[s]["overall"]) for s in PAIR), _pct(int(data["x_wins"]), k),
                     _pct(int(data["y_wins"]), k)])
    lines += _table(["судья", "оценок", *(f"{s}: побед" for s in PAIR), "ничьих",
                     *(f"{s}: среднее" for s in PAIR), "X побед", "Y побед"], body)
    agreement = cast(dict[str, Any], summary["agreement"])
    majority = cast(dict[str, int], agreement["majority"])
    lines += ["", "## Согласие судей", "",
              f"- Вопросов с оценками двух и более судей: {agreement['questions_multi']}; все выбрали одно и то же: "
              f"{_pct(int(agreement['unanimous']), int(agreement['questions_multi']))}.",
              "- Победитель по большинству судей: " + ", ".join(
                  f"{k} {v}" for k, v in sorted(majority.items())) + "." if majority else "- Оценок нет."]
    problems = cast(list[str], summary["problems"])
    if problems:
        lines += ["", "## Что не засчитано", ""] + [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.unblind", description="Свести оценки слепых судей")
    parser.add_argument("run", help="имя прогона в eval/runs или путь к его папке")
    parser.add_argument("--scores", type=Path, nargs="*", default=None,
                        help="файлы оценок (по умолчанию blind/scores/*.json)")
    parser.add_argument("--set", type=Path, default=None, help="набор вопросов (по умолчанию — из meta.json)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]
    run_dir = resolve_run_dir(args.run)
    blind = run_dir / "blind"
    key_path = blind / "key.json"
    if not key_path.is_file():
        print(f"нет ключа {key_path}: сначала python -m eval.blind {args.run}", file=sys.stderr)
        return 2
    files = list(args.scores) if args.scores else sorted((blind / "scores").glob("*.json"))
    if not files:
        print(f"нет файлов оценок в {blind / 'scores'}", file=sys.stderr)
        return 2
    try:
        items = run_items(run_dir, args.set)
    except DatasetError as exc:
        print(exc, file=sys.stderr)
        return 2
    collected = collect(files, cast(dict[str, Any], read_json(key_path)), items)
    summary = summarize(collected)
    text = render(summary, run_dir.name)
    write_json(blind / "unblind.json", summary)
    (blind / "unblind.md").write_text(text, encoding="utf-8")
    print(text)
    return 0 if collected.ratings else 1


if __name__ == "__main__":
    sys.exit(main())
