---
id: L1.3
title: "Функции и модули"
module: 1
topic: python-functions
resource_type: lecture
level: all
---

# Функции и модули

## О чём лекция

Функция — именованный кусок кода: принимает данные, делает работу и возвращает результат. Модуль — файл с функциями,
который можно подключить через `import`. Вместе они избавляют от копирования одного и того же кода. Лекция
опирается на L1.1 (типы) и L1.2 (условия и циклы). В hw1 «Анализ списка покупок» нужно написать не меньше трёх
своих функций (guide-1, критерии оценки).

## Объявление функции

```python
def line_total(quantity, price):
    """Вернуть стоимость позиции: количество, умноженное на цену."""
    return quantity * price

print(line_total(2, 89.9))  # 179.8
```

Строка с `def` содержит имя, параметры в скобках и двоеточие, тело пишется с отступом. Первая строка в тройных
кавычках — docstring, описание функции: его показывает `help(line_total)`. `return` завершает функцию и отдаёт
значение. Если `return` нет, функция возвращает `None`.

```python
def show_total(total):
    print(f"Итого: {total:.2f}")

result = show_total(100)  # напечатает: Итого: 100.00
print(result)             # None
```

Это главное отличие `print` от `return`: `print` только показывает значение на экране, а `return` позволяет
использовать результат дальше, например сложить или сохранить.

## Параметры и аргументы

Аргументы передают по позиции или по имени. У параметра может быть значение по умолчанию.

```python
def apply_discount(total, percent=5, min_total=1000):
    """Скидка в процентах, если сумма не меньше min_total."""
    if total >= min_total:
        return round(total * (1 - percent / 100), 2)
    return total

print(apply_discount(1500))                # 1425.0
print(apply_discount(1500, 10))            # 1350.0
print(apply_discount(800, min_total=500))  # 760.0
```

Именованные аргументы пишутся после позиционных. Функция может вернуть несколько значений сразу, на деле это
кортеж, который удобно распаковать:

```python
def min_max(values):
    return min(values), max(values)

low, high = min_max([89.9, 54.0, 412.5])
print(low, high)  # 54.0 412.5
```

## Область видимости

Переменные, созданные внутри функции, локальные: снаружи их не видно. Читать глобальную переменную функция может,
но такой код трудно проверять. Надёжнее передавать всё нужное через параметры и возвращать результат.

```python
rate = 0.9

def discounted(price):
    new_price = price * rate  # rate берётся из глобальной области
    return new_price

print(discounted(100))  # 90.0
print(new_price)        # NameError: name 'new_price' is not defined
```

Лучше так: `def discounted(price, rate=0.9):`. Тогда функция зависит только от своих аргументов.

## Функция как аргумент: sorted с key

Функцию можно передать в другую функцию. Частый случай — сортировка по выбранному полю. `lambda` — короткая
безымянная функция в одну строку.

```python
purchases = [("сыр", 412.5), ("хлеб", 54.0), ("молоко", 179.8)]
by_amount = sorted(purchases, key=lambda p: p[1], reverse=True)
print(by_amount[:2])  # [('сыр', 412.5), ('молоко', 179.8)]
```

## Модули стандартной библиотеки

Python поставляется с большой стандартной библиотекой, её модули ничего устанавливать не требуют.

```python
import math
from statistics import mean, median
from datetime import date

prices = [89.9, 54.0, 412.5, 79.9]
print(mean(prices))          # 159.075
print(median(prices))        # 84.9
print(math.ceil(159.075))    # 160
print(date(2026, 11, 15) - date(2026, 11, 2))  # 13 days, 0:00:00
```

Варианты импорта: `import math` (обращение `math.sqrt(16)`), `from math import sqrt` (просто `sqrt(16)`),
`import statistics as st` (короткий псевдоним). Запись `from math import *` лучше не использовать: непонятно,
откуда взялось имя, и можно случайно перекрыть свои переменные.

Для CSV есть модуль `csv`. `DictReader` превращает каждую строку файла в словарь с ключами из заголовка:

```python
import csv

with open("shopping_list.csv", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

print(rows[0]["item"], rows[0]["price"])  # все значения прочитаны как строки
```

Документация модуля `csv` советует открывать файл с `newline=""`, а `encoding="utf-8"` нужен для кириллицы.

## Свой модуль

Модуль — это просто файл `.py`. Например, `shop_utils.py`:

```python
"""Помощники для списка покупок."""


def parse_price(text):
    """Превратить строку вида '94,50' или '94.50' в число."""
    return float(text.replace(",", "."))


if __name__ == "__main__":
    print(parse_price("94,50"))  # 94.5, только при запуске файла напрямую
```

В ноутбуке модуль подключается так же, как стандартный: `import shop_utils`, затем
`shop_utils.parse_price("12,5")`. В Colab файл можно создать ячейкой, первая строка которой `%%writefile shop_utils.py`,
или загрузить через панель «Файлы». Python ищет модули в текущей папке и в путях из `sys.path`.

Модуль загружается один раз за сеанс. Если вы поменяли файл, перезапустите среду выполнения или выполните
`importlib.reload(shop_utils)` после `import importlib`.

Сторонние пакеты ставят через pip, в Colab командой `%pip install <пакет>`. pandas, matplotlib и seaborn в Colab уже
установлены.

## Частые ошибки

- **Нет скобок при вызове**: `print(line_total)` печатает `<function line_total at 0x...>` вместо результата.
- **`print` вместо `return`**: функция возвращает `None`, и следующий расчёт падает с
  `TypeError: unsupported operand type(s) for +: 'NoneType' and 'float'`.
- **Изменяемое значение по умолчанию**: в `def add_item(item, basket=[])` список создаётся один раз при объявлении
  функции и копится между вызовами. Пишите `basket=None`, а внутри `if basket is None: basket = []`.
- **Не тот набор аргументов**: `TypeError: line_total() missing 1 required positional argument: 'price'`.
- **Ячейка с функцией не выполнена**: в ноутбуке вызов до выполнения ячейки с `def` даёт `NameError`.
- **Свой файл назван как стандартный модуль** (`csv.py`, `random.py`, `statistics.py`): он перекрывает настоящий
  модуль, и появляются странные `AttributeError` и `ImportError`.
- **`ModuleNotFoundError`**: файла модуля нет в текущей папке или пакет не установлен.

## Если сложно

- Повторите L1.2: тело функции почти всегда состоит из условий и циклов, и если цикл не работает сам по себе,
  функция тоже не заработает. Типы и `None` — в L1.1.
- Делайте функции из готового кода: возьмите работающий цикл, оберните в `def`, конкретные значения замените
  параметрами, результат верните через `return`.
- Проверяйте каждую функцию на маленьком примере, ответ которого знаете заранее: `line_total(2, 10)` должно
  вернуть `20`.
- Шаги hw1 (guide-1) подсказывают, какие функции нужны. Записи вебинаров появляются в LMS в течение 24 часов.

## Для продвинутых

- Аннотации типов: `def line_total(quantity: float, price: float) -> float:`, коллекции вида
  `list[dict[str, str]]` (Python 3.9+), `float | None` (Python 3.10+). Проверка — mypy или basedpyright.
- `*args` и `**kwargs`, параметры только по имени (после `*`) и только по позиции (до `/`, Python 3.8+):
  https://docs.python.org/3/tutorial/controlflow.html#more-on-defining-functions
- Пакеты: папка с модулями и `__init__.py`, относительный импорт, `pyproject.toml`:
  https://docs.python.org/3/tutorial/modules.html
- Генераторы с `yield` для потоковой обработки больших файлов и `functools.lru_cache` для кеширования.
- Тесты для своих функций: `assert` и pytest. Сортировка с `key`: https://docs.python.org/3/howto/sorting.html
