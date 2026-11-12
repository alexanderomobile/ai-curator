---
id: L3.1
title: "Matplotlib: основы графиков"
module: 3
topic: matplotlib-basics
resource_type: lecture
level: all
---

# Matplotlib: основы графиков

## О чём лекция

Matplotlib — базовая библиотека визуализации в Python. На ней построены seaborn (лекция L3.2) и метод
`DataFrame.plot` в pandas, поэтому понимать её устройство полезно, даже если вы рисуете через другие инструменты.
В лекции разберём, из чего состоит график, четыре базовых типа графиков, подписи, несколько графиков на одной
картинке и сохранение в файл. Примеры проверены на Python 3.12, matplotlib 3.x и pandas 2.x и запускаются
в Google Colab или Jupyter.

## Figure и Axes: из чего состоит график

У каждого графика в Matplotlib два главных объекта:

- **Figure** — «холст», весь рисунок целиком. На одном холсте может быть несколько графиков.
- **Axes** — одна область с осями внутри холста, то есть сам график. У Axes есть оси X и Y, заголовок,
  подписи и легенда.

Рисовать можно двумя способами. Неявный способ — функции `plt.plot()`, `plt.title()`: они работают с «текущим»
графиком, который Matplotlib выбирает сам. Явный (объектный) способ — сначала получить `fig` и `ax`, а потом вызывать
методы `ax`. В курсе используем явный способ: с ним всегда видно, на какой график попадает каждая команда.

```python
import matplotlib.pyplot as plt

months = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн"]
revenue = [410, 385, 450, 470, 520, 610]  # выручка, тыс. руб.

fig, ax = plt.subplots(figsize=(8, 4))  # ширина и высота в дюймах
ax.plot(months, revenue, marker="o")
ax.set_title("Выручка за первое полугодие")
ax.set_xlabel("Месяц")
ax.set_ylabel("Выручка, тыс. руб.")
ax.grid(alpha=0.3)
plt.show()
```

`plt.subplots()` возвращает пару: холст и график. В Jupyter и Colab график появится под ячейкой и без
`plt.show()`, а в обычном `.py`-скрипте без этого вызова окно с графиком не откроется.

## Четыре базовых типа графиков

| Метод | Тип | Когда использовать |
|---|---|---|
| `ax.plot(x, y)` | линия | изменение во времени |
| `ax.bar(x, h)`, `ax.barh(y, w)` | столбцы | сравнение категорий |
| `ax.hist(values, bins=20)` | гистограмма | распределение одного числового признака |
| `ax.scatter(x, y)` | точки | связь двух числовых признаков |

Пример: горизонтальные столбцы с подписями значений и гистограмма на случайных данных.

```python
import numpy as np

categories = ["Кухня", "Текстиль", "Декор", "Хранение"]
sales = [1240, 860, 530, 410]  # тыс. руб.

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(categories, sales)
ax.bar_label(bars, fmt="%d", padding=3)  # подписи значений у столбцов
ax.invert_yaxis()                        # первая категория сверху
ax.set_xlabel("Выручка, тыс. руб.")
plt.show()

rng = np.random.default_rng(42)
order_amounts = rng.normal(loc=2500, scale=700, size=500)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(order_amounts, bins=25, edgecolor="white")
ax.set_xlabel("Сумма заказа, руб.")
ax.set_ylabel("Число заказов")
plt.show()
```

У `barh` первая категория по умолчанию оказывается внизу, поэтому вызываем `ax.invert_yaxis()`. Метод
`bar_label` есть в Matplotlib начиная с версии 3.4. Параметр `bins` задаёт число интервалов гистограммы: при слишком
малом числе форма распределения теряется, при слишком большом появляется «шум».

## Графики из pandas

Методы `DataFrame.plot()` и `Series.plot()` рисуют через Matplotlib и возвращают тот же объект Axes. Передайте
им `ax=ax`, чтобы график попал в нужную область, а подписи добавляйте обычными методами `ax`.

```python
import pandas as pd

df = pd.DataFrame({
    "month": pd.date_range("2025-01-01", periods=6, freq="MS"),  # MS = начало месяца
    "site": [250, 240, 270, 290, 310, 360],
    "app": [160, 145, 180, 180, 210, 250],
})

fig, ax = plt.subplots(figsize=(8, 4))
df.plot(x="month", y=["site", "app"], ax=ax, marker="o")
ax.legend(["Сайт", "Приложение"])
ax.set_xlabel("")
ax.set_ylabel("Выручка, тыс. руб.")
plt.show()
```

Обычно данные сначала агрегируют (`groupby`, лекция L2.3), а потом рисуют уже готовую короткую таблицу.

## Несколько графиков на одной фигуре

`plt.subplots(строки, столбцы)` создаёт сетку графиков. Если графиков больше одного, вторым значением
возвращается массив Axes, к элементам которого обращаются по индексу.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(months, revenue, marker="o")
axes[0].set_title("Выручка по месяцам, тыс. руб.")
axes[1].barh(categories, sales)
axes[1].invert_yaxis()
axes[1].set_title("Выручка по категориям, тыс. руб.")
fig.suptitle("Магазин «Полка», январь-июнь 2025")
fig.tight_layout()  # раздвигает графики, чтобы подписи не наезжали
plt.show()
```

В сетке 2×2 массив двумерный: `axes[0, 0]`, `axes[1, 1]`. Чтобы обойти все графики в цикле,
используйте `for ax in axes.flat:`. Параметры `sharex=True` и `sharey=True` дают графикам общую шкалу.

## Сохранение в файл

```python
fig.savefig("chart.png", dpi=150, bbox_inches="tight")
```

Формат определяется по расширению: `.png`, `.svg`, `.pdf`. `dpi` — плотность точек (150-200 достаточно для отчёта),
`bbox_inches="tight"` обрезает лишние поля. Сохраняйте через `fig.savefig` и до `plt.show()`. В Colab файл
попадает во временную папку `/content`: скачайте его через панель «Файлы», иначе он пропадёт, когда среда
выполнения отключится (например, после долгого простоя).

## Частые ошибки

- **`AttributeError: 'numpy.ndarray' object has no attribute 'plot'`** — после `plt.subplots(1, 2)` в переменной
  лежит массив графиков, а не один график. Обращайтесь к `axes[0]`, `axes[1]`.
- **`TypeError: 'Text' object is not callable`** — написали `ax.title("...")`. У Axes методы с приставкой `set_`:
  `set_title`, `set_xlabel`, `set_ylabel`. Без приставки они есть только у `plt`: `plt.title`, `plt.xlabel`.
  Вызов `ax.xlabel(...)` даёт `AttributeError: 'Axes' object has no attribute 'xlabel'`.
- **Пустой PNG** — `plt.savefig()` вызван после `plt.show()` или в следующей ячейке Jupyter: текущая фигура
  к этому моменту закрыта, и сохраняется новая, пустая. Вызывайте `fig.savefig(...)` в той же ячейке до показа.
- **Подписи наезжают друг на друга** — добавьте `fig.tight_layout()` или создайте фигуру с
  `plt.subplots(..., layout="constrained")`. Длинные названия категорий лучше выводить на `barh`.
- **Даты на оси X слиплись** — `fig.autofmt_xdate()` повернёт и выровняет подписи дат.
- **Предупреждение `More than 20 figures have been opened`** — графики создаются в цикле и не закрываются.
  Закрывайте ненужные фигуры через `plt.close(fig)`.
- **Нет единиц измерения** — «Выручка» без «тыс. руб.». Python не сообщит об ошибке, но в ДЗ это снижает баллы
  за оформление (методичка guide-3).

## Если сложно

- Списки, которые мы передаём в `plot` и `bar`, разобраны в L1.1 «Переменные и типы данных», циклы `for` —
  в L1.2 «Условия и циклы».
- Импорт `import matplotlib.pyplot as plt` и вызов методов объектов повторите по L1.3 «Функции и модули».
- Как выбрать столбец DataFrame и прочитать CSV — L2.1; как посчитать суммы по группам перед графиком — L2.3.
- Начните с шаблона из раздела «Figure и Axes»: `fig, ax = plt.subplots()`, одна команда рисования, затем
  `set_title`, `set_xlabel`, `set_ylabel`. Когда он станет привычным, переходите к сетке графиков.

## Для продвинутых

- Сложная компоновка: `plt.subplot_mosaic([["a", "b"], ["c", "c"]])` и `GridSpec` — графики разного размера
  на одном холсте.
- Форматирование осей через `matplotlib.ticker`: `PercentFormatter`, `MaxNLocator`,
  `ax.yaxis.set_major_formatter("{x:,.0f}")`; для дат — `matplotlib.dates.ConciseDateFormatter`.
- Стили и настройки: `plt.style.available`, `plt.style.use(...)`, словарь `rcParams` и временные настройки
  через `with plt.rc_context({...}):`.
- Документация: раздел Quick start guide и галерея примеров на matplotlib.org.
