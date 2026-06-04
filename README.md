# Linear Models from Scratch

Реализация линейной и логистической регрессии с нуля на Python без использования готовых ML-алгоритмов из sklearn.

## О проекте

Цель — показать понимание математики за алгоритмами: градиентный спуск, функции потерь, регуляризация. Sklearn используется только для загрузки датасетов и сравнения результатов.

**Результаты:**
- LinearRegression: R² = 0.455 (sklearn: 0.453) — разница 0.002
- LogisticRegression: Accuracy = 0.956, ROC AUC = 0.996 (sklearn: 0.983 / 0.995)

## Стек

Python · NumPy · scikit-learn (метрики и датасеты) · Jupyter · pytest · ruff · GitLab CI · uv

## Структура проекта

```
├── src/
│   ├── linearModel.py          # Абстрактный базовый класс (ABC)
│   ├── linearRegression.py     # Линейная регрессия (MSE, градиентный спуск)
│   └── logisticRegression.py   # Логистическая регрессия (кросс-энтропия, сигмоида)
├── notebooks/
│   └── test.ipynb              # Сравнение с sklearn, визуализации, auto_tune
├── tests/                      # Автотесты (pytest)
├── .gitlab-ci.yml              # CI: линтер + тесты при каждом пуше
└── pyproject.toml              # Зависимости (uv)
```

## Что реализовано

**Базовый класс `LinearModel`** (абстрактный):
- Градиентный спуск с настраиваемым `learning_rate` и `n_iterations`
- История потерь `loss_history` для отслеживания обучения
- Абстрактные методы: `_forward`, `_compute_loss`, `_compute_gradients`, `_compute_metric`

**`LinearRegression`**:
- Функция потерь: MSE
- Градиенты: аналитически из MSE
- Метрика: R²

**`LogisticRegression`**:
- Активация: сигмоида с защитой от переполнения (`np.clip`)
- Функция потерь: бинарная кросс-энтропия
- Отдельные методы `predict` (классы) и `predict_proba` (вероятности)
- `stratify` при разбивке данных в `auto_tune`

**`auto_tune`** (у обоих классов):
- Перебор `learning_rate` и `n_iterations` по сетке
- Валидационная выборка, возврат лучших параметров

## Запуск

```bash
# Установка зависимостей
uv venv && source .venv/bin/activate
uv sync

# Линтер
uvx ruff check --fix && uvx ruff format

# Тесты
uv run pytest

# Ноутбук
jupyter lab notebooks/test.ipynb
```

## Датасеты

- **Diabetes** (sklearn) — регрессия, 442 объекта, 10 признаков
- **Breast Cancer** (sklearn) — бинарная классификация, 569 объектов, 30 признаков

---

*Проект выполнен в рамках подготовки к DS/ML позиции. Код проходил code review и CI-проверки.*
