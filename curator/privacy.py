"""Обезличивание: код сессии вместо идентификатора и вычистка контактов из текста вопроса.

В журнал идёт не идентификатор сессии из браузера, а его HMAC с секретной солью. Из вопроса вырезаются почта,
ники и ссылки Telegram, телефоны и длинные номера, номера документов и имя после «меня зовут». Вопрос вычищается
ещё до отправки в OpenAI: модели номер телефона студента не нужен. Публичные контакты курса (почта поддержки)
остаются, иначе вопрос «что за адрес support@...?» потерял бы смысл.

Подход и правила — из FAQ-ассистента (PEcf11, backend/privacy.py), проверенные там на живых вопросах.
Имя после «меня зовут» вырезается и со строчной буквы («меня зовут иван петров»): в чате часто пишут без заглавных.
Внутри кода (`…` и ```…```) длинные числа не трогаются — это вывод программ, — но мобильный номер вида
+7 9xx … / 8 9xx … вырезается и там. Код студенты часто вставляют и без обратных кавычек, поэтому и вне кода
не трогаются даты ISO («pd.to_datetime('2026-11-12')»), число строк данных («1200000 строк»), переменная
pandas.query после оператора («qty >= @limit») и pandas Series по-русски («Серия из 1000 строк»).
Ограничение: имя без подсказки («Иван Петров, когда дедлайн?») правилами не распознать — такие записи покрывает
срок хранения журнала (90 дней).
"""
from __future__ import annotations

import hashlib
import hmac
import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_TG_LINK = re.compile(r"(?i)(?:https?://)?(?:t\.me|telegram\.me)/[A-Za-z0-9_+]{4,}")
# Ник после кириллицы («тг@ivan_petrov») тоже ник; после латиницы и точки это часть адреса почты.
_MENTION = re.compile(r"(?<![A-Za-z0-9_.])@[A-Za-z][A-Za-z0-9_]{3,}")
# Перед @ — оператор сравнения или присваивания, «in»: это переменная в строке pandas.query («qty >= @limit»,
# «city in @cities»), а не ник.
_QUERY_VARIABLE_BEFORE = re.compile(r"(?:[=<>]|\bin)\s*$")
# Паспорт серия/номер: «4510 123456», «45 10 № 123456», «серия 4510 номер 123456». Слово «серия» — только с цифрами
# паспорта сразу после него: «Серия из 1000 строк, 250 пропусков» — pandas Series, а не документ.
_DOCUMENT = re.compile(r"(?i)(?:паспорт\w*|снилс)[^\d\n]{0,15}\d{2,3}[\s-]?\d{2,3}[^\d\n]{0,12}\d{3,6}"
                       r"|серия\s*№?\s*\d{2}\s?\d{2}[^\d\n]{0,12}\d{6}"
                       r"|\d{4}\s*№\s*\d{6}")
# Имя — только после явной подсказки: 1-3 слова. С заглавной буквы — как написано; со строчной — если первое слово
# не служебное («меня зовут не помню как» — не имя). У имени с заглавной следующие слова тоже с заглавной.
_NAME_HINT = re.compile(r"(меня\s+зовут|мо[её]\s+имя|моя\s+фамилия)(\s*[:—–-]?\s*)", re.IGNORECASE)
_NAME_WORD = re.compile(r"[А-ЯЁA-Zа-яёa-z][а-яёa-z]*(?:-[А-ЯЁA-Zа-яёa-z][а-яёa-z]*)?")
_NOT_NAME = frozenset(
    "не ни как так что это вот тут здесь там просто уже все всё и а но или да нет же ли бы по на в во с со у к ко "
    "о об от до из за для при про я мне меня мой моя моё мое мои вы вам вас ваш ваша он она они его её ее их "
    "когда где куда зачем почему если чтобы хочу хотел хотела хотим подскажите подскажи скажите скажи помогите "
    "помоги спросить вопрос есть был была были буду будет студент студентка учусь тоже теперь сейчас".split())
_MOBILE = re.compile(r"(?<![\d\w])(?:\+7|8)[\s(-]*9\d{2}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)")
# 7+ цифр с пробелами, дефисами, точками, скобками и плюсом: телефоны («8.900.123.45.67»), карты, счета.
_LONG_NUMBER = re.compile(r"\+?\d[\d\s().\-]{5,}\d")
# Дата в начале такого числа («15.11.2026», «15.11.2026 23») — не номер, если после неё мало цифр.
_DATE = re.compile(r"\d{1,2}\.\d{1,2}\.(?:\d{4}|\d{2})")
# Даты ISO из кода и данных курса («2026-11-12», «2026-11-01 2026-11-15», «2026-11-12 14» перед «:00») — не номер.
_ISO_DATES = re.compile(r"(?:(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?:[ T]\d{1,2})?[\s,;]*)+")
# Суммы с разрядами («1 500 000 рублей») и годы («2019-2026») номерами не считаются. Количество строк данных —
# и без разрядов («1200000 строк», «250000 пропусков»): деньги без разрядов («4276123456789012 руб») — номер.
_AMOUNT_AFTER = re.compile(r"\s*(?:руб|₽|\$|usd|долл|евро|eur|€|млн|тыс|строк|заказ|значени|пропуск|записей)",
                           re.IGNORECASE)
_COUNT_AFTER = re.compile(r"\s*(?:строк|заказ|значени|пропуск|записей)", re.IGNORECASE)
_GROUPED = re.compile(r"\d{1,3}(?:[  .]\d{3})+")
_YEARS = re.compile(r"(?:(?:19|20)\d{2})(?:\s*[-–]\s*|\s+)(?:(?:19|20)\d{2})(?:(?:\s*[-–]\s*|\s+)(?:19|20)\d{2})*")
_DECIMAL = re.compile(r"-?\d+\.\d+")
_CODE = re.compile(r"```[\s\S]*?(?:```|$)|`[^`\n]+`")


def user_hash(salt: str, raw_id: object) -> str:
    """Обезличенный код сессии: по нему видно реплики одного диалога, но не человека."""
    if raw_id is None or raw_id == "":
        return ""
    digest = hmac.new(salt.encode("utf-8"), str(raw_id).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:16]


def _redact_names(text: str) -> str:
    out: list[str] = []
    last = 0
    for hint in _NAME_HINT.finditer(text):
        if hint.start() < last:
            continue
        pos = hint.end()
        words: list[re.Match[str]] = []
        while len(words) < 3:
            if words:  # следующее слово — через пробелы, без знаков препинания
                gap = re.match(r"[ \t]+", text[pos:])
                if not gap:
                    break
                candidate = _NAME_WORD.match(text, pos + gap.end())
            else:
                candidate = _NAME_WORD.match(text, pos)
            if candidate is None or candidate.group(0).lower() in _NOT_NAME:
                break
            if words and words[0].group(0)[0].isupper() and not candidate.group(0)[0].isupper():
                break
            words.append(candidate)
            pos = candidate.end()
        if not words:
            continue
        out.append(text[last:hint.end()].rstrip() + " [имя]")
        last = words[-1].end()
    out.append(text[last:])
    return "".join(out)


def redact(text: str, keep: tuple[str, ...] = ()) -> str:
    """Убирает контакты и номера. keep — публичные контакты курса, их оставляем."""
    kept = {k.lower() for k in keep}

    def _unless_kept(placeholder: str):
        return lambda m: m.group(0) if m.group(0).lower() in kept else placeholder

    def _mention(match: re.Match[str]) -> str:
        line_start = match.string.rfind("\n", 0, match.start()) + 1
        if _QUERY_VARIABLE_BEFORE.search(match.string, line_start, match.start()):
            return match.group(0)
        return match.group(0) if match.group(0).lower() in kept else "[ник]"

    text = _EMAIL.sub(_unless_kept("[почта]"), text)
    text = _TG_LINK.sub("[ник]", text)
    text = _MENTION.sub(_mention, text)
    text = _DOCUMENT.sub("[документ]", text)
    text = _redact_names(text)

    def _numbers(part: str) -> str:
        def _number(match: re.Match[str]) -> str:
            value = match.group(0)
            stripped = value.strip()
            if sum(ch.isdigit() for ch in value) < 7:
                return value
            if _DECIMAL.fullmatch(stripped) or _YEARS.fullmatch(stripped) or _ISO_DATES.fullmatch(stripped):
                return value
            date = _DATE.match(stripped)
            if date and sum(ch.isdigit() for ch in stripped[date.end():]) < 7:
                return value
            if _GROUPED.fullmatch(stripped) and _AMOUNT_AFTER.match(part, match.end()):
                return value
            if stripped.isdigit() and _COUNT_AFTER.match(part, match.end()):
                return value
            return "[номер]"

        return _LONG_NUMBER.sub(_number, part)

    # Код и вывод программы («0.30000000000000004», номера заказов в примере) номерами людей не бывают — кроме
    # мобильного номера, его в коде тоже вырезаем.
    pieces: list[str] = []
    last = 0
    for code in _CODE.finditer(text):
        pieces.append(_numbers(text[last:code.start()]))
        pieces.append(_MOBILE.sub("[номер]", code.group(0)))
        last = code.end()
    pieces.append(_numbers(text[last:]))
    return "".join(pieces)
