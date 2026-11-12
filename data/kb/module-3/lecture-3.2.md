---
id: L3.2
title: "Seaborn и выбор типа графика"
module: 3
topic: seaborn-chart-choice
resource_type: lecture
level: all
---

# Seaborn и выбор типа графика

## О чём лекция

Seaborn — надстройка над Matplotlib для статистических графиков. Она берёт DataFrame целиком, сама группирует
строки, считает средние и интервалы, раскрашивает данные по категориям и добавляет легенду. В лекции разберём, как
устроен вызов seaborn, чем функции уровня графика отличаются от функций уровня фигуры, что на самом деле показывает
`barplot` и как выбрать тип графика под вопрос. Примеры проверены на seaborn 0.13, matplotlib 3.x и pandas 2.x.

## Как устроен вызов seaborn

Почти все функции вызываются по одному шаблону: `sns.функция(data=df, x="колонка", y="колонка", hue="колонка")`.
Параметр `hue` раскрашивает данные по значениям ещё одной колонки. Данные должны быть в «длинном» формате: одна
строка — одно наблюдение (например, один заказ), колонки — его признаки.

Подготовим вымышленные заказы магазина, на них построим все примеры лекции:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")  # тема оформления для всех следующих графиков

rng = np.random.default_rng(7)
n = 600
orders = pd.DataFrame({
    "channel": rng.choice(["Сайт", "Приложение", "Маркетплейс"], size=n, p=[0.5, 0.3, 0.2]),
    "category": rng.choice(["Кухня", "Текстиль", "Декор", "Хранение"], size=n),
    "items": rng.integers(1, 6, size=n),
})
orders["amount"] = (orders["items"] * rng.normal(900, 250, size=n)).clip(lower=150).round()
```

## Функции уровня графика и уровня фигуры

**Уровень графика (axes-level):** `histplot`, `kdeplot`, `boxplot`, `violinplot`, `barplot`, `countplot`,
`scatterplot`, `lineplot`, `heatmap`. Они рисуют на одном Axes, принимают `ax=` и возвращают Axes, поэтому их
можно разложить по сетке `plt.subplots` из лекции L3.1 и собрать дашборд.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data=orders, x="amount", bins=30, ax=axes[0])
axes[0].set(title="Распределение суммы заказа", xlabel="Сумма заказа, руб.", ylabel="Число заказов")
sns.boxplot(data=orders, x="channel", y="amount", ax=axes[1])
axes[1].set(title="Сумма заказа по каналам", xlabel="", ylabel="Сумма заказа, руб.")
fig.tight_layout()
plt.show()
```

**Уровень фигуры (figure-level):** `displot`, `catplot`, `relplot`, `lmplot`, `pairplot`, `jointplot`. Они создают
собственную фигуру, параметр `ax=` не принимают и возвращают объект-сетку (`FacetGrid`, у `pairplot` — `PairGrid`,
у `jointplot` — `JointGrid`). Их сила — «малые множества»: один и тот же график для каждой категории через `col=`
или `row=`.

```python
g = sns.displot(data=orders, x="amount", col="channel", bins=20, height=3.5)
g.set_axis_labels("Сумма заказа, руб.", "Число заказов")
g.set_titles("{col_name}")
plt.show()
```

## Что показывает barplot: среднее, а не сумму

По умолчанию `sns.barplot` считает **среднее** по каждой категории и рисует чёрную черту — 95-процентный
доверительный интервал, полученный бутстрепом. Для выручки обычно нужна сумма:

```python
order = orders.groupby("category")["amount"].sum().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=orders, x="amount", y="category", estimator="sum", errorbar=None,
            order=order, color="tab:blue", ax=ax)
ax.set(xlabel="Выручка, руб.", ylabel="")
plt.show()
```

Параметр `errorbar` появился в версии 0.12 вместо устаревшего `ci`. Функция `countplot` считает число строк в каждой
категории, а `lineplot` при повторяющихся значениях X тоже усредняет их и рисует доверительный интервал. Если таблица
уже агрегирована (одна строка на категорию), интервал не строится.

## Как выбрать тип графика

Сначала сформулируйте вопрос, потом выбирайте форму:

| Вопрос | Тип графика | Функции |
|---|---|---|
| Как меняется во времени? | линия | `sns.lineplot`, `ax.plot` |
| Какая категория больше? | столбцы, отсортированные; горизонтальные при длинных названиях | `sns.barplot`, `ax.barh` |
| Как распределены значения? | гистограмма, «ящик с усами», «скрипка» | `sns.histplot`, `sns.boxplot`, `sns.violinplot` |
| Связаны ли два числа? | точки, линия регрессии | `sns.scatterplot`, `sns.regplot` |
| Из каких частей целое? | составные столбцы; круговая — только для 2-3 частей | `df.plot(kind="bar", stacked=True)` |
| Значение на пересечении двух категорий | тепловая карта | `sns.heatmap` |

Почему столбцы лучше круговой диаграммы: в экспериментах У. Кливленда и Р. Макгилла (1984) люди точнее
сравнивали значения по положению на общей шкале, как у столбцов, чем по углам секторов, как у круговой диаграммы.

«Ящик с усами» читается так: ящик — межквартильный размах (от Q1 до Q3), линия внутри — медиана, «усы» доходят
до самой дальней точки в пределах 1,5 межквартильного размаха от ящика, точки за ними — выбросы.

Тепловая карта удобна для таблицы «категория × канал»:

```python
pivot = orders.pivot_table(index="category", columns="channel", values="amount", aggfunc="sum")

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(pivot / 1000, annot=True, fmt=".0f", cmap="Blues",
            cbar_kws={"label": "Выручка, тыс. руб."}, ax=ax)
ax.set(xlabel="", ylabel="")
ax.tick_params(axis="y", rotation=0)  # названия категорий горизонтально
plt.show()
```

## Частые ошибки

- **Среднее вместо суммы.** `barplot` без `estimator="sum"` показывает средний чек, а не выручку. Проверьте
  результат через `groupby(...).sum()`.
- **`ax=` в функции уровня фигуры.** `sns.catplot(..., ax=ax)` выдаёт `UserWarning: catplot is a figure-level
  function and does not accept target axes` и рисует отдельную фигуру. Для дашборда берите `boxplot`, `barplot`
  и другие функции уровня графика.
- **`palette` без `hue`.** В 0.13 это `FutureWarning`: так перестанет работать в 0.14. Задайте один цвет через
  `color=` или укажите ту же колонку в `hue=` и добавьте `legend=False`.
- **Широкая таблица вместо длинной.** Если месяцы или каналы стоят в колонках, переведите таблицу в длинный формат:
  `df.melt(id_vars="month", var_name="channel", value_name="revenue")`.
- **Месяцы по алфавиту.** После `groupby` по строковым названиям месяцев строки сортируются по алфавиту
  («Авг», «Апр», «Дек»...). Группируйте по датам или задайте порядок через `order=`.
- **Радужная палитра для упорядоченных значений.** Для «мало → много» берите последовательную палитру
  (`"Blues"`), для категорий — `"colorblind"`.
- **Предупреждение — не ошибка.** Например, при seaborn 0.13.2 и Matplotlib 3.11 `boxplot` выводит
  `MatplotlibDeprecationWarning` про параметр `vert`: это несовместимость версий библиотек, график строится верно.

## Если сложно

- Figure, Axes и сетку `plt.subplots` повторите по L3.1 «Matplotlib: основы графиков».
- Как посчитать суммы по группам (`groupby`, `agg`, `pivot_table`) — L2.3 «Группировка и агрегация».
- Seaborn молча пропускает пустые значения: перед графиком проверьте их по L2.2 «Фильтрация, сортировка, пропуски».
- Для ДЗ достаточно трёх функций: `histplot`, `barplot` с `estimator="sum"` и `lineplot`. Начните с них.

## Для продвинутых

- Интерфейс `seaborn.objects` (с версии 0.12): `so.Plot(orders, x="channel", y="amount").add(so.Bar(), so.Agg("sum"))`.
- Варианты `errorbar`: `("ci", 95)`, `"sd"`, `"se"`, `"pi"` — раздел Statistical estimation and error bars
  в документации seaborn.
- Разведочный анализ: `pairplot`, `jointplot`, `ecdfplot`; собственные графики в сетке через
  `FacetGrid.map_dataframe`.
- Первоисточник по восприятию графиков: Cleveland W. S., McGill R. Graphical Perception (1984).
