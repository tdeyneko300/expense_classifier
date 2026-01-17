# Pet project: Expense classifier (RU)

Проект обучает multi‑class модель (например: `food`, `transport`, `subscriptions`, `home`, …) и даёт CLI, который принимает одну операцию и возвращает:

- `category` — предсказанная категория
- `confidence` — уверенность (вероятность для модели)

## Что внутри

- Бейзлайн: TF‑IDF для `text` и `merchant` + линейная модель.
- Модели:
	- `logreg` (по умолчанию) — Logistic Regression
	- `svm` — Linear SVM + калибровка вероятностей (чтобы выдавать `confidence`)
- CLI команды:
	- `train` — обучить и сохранить модель в `.joblib`
	- `classify` — классифицировать одну операцию

## Быстрый старт (Windows PowerShell)

```powershell
cd E:\txData\pet_expense_classifier

# 1) Виртуальное окружение
python -m venv .venv
\.\.venv\Scripts\Activate.ps1

# 2) Зависимости
pip install -r requirements.txt

# 3) Обучение (на синтетических данных из репозитория)
python .\scripts\train.py --data .\data\transactions_labeled.csv --out .\model.joblib

# 4) Предсказание категории
python -m expense_classifier.cli classify --model .\model.joblib --text "Yandex Go" --merchant "Yandex Go" --amount 420 --topk 3
```

## Формат данных (CSV)

Нужен CSV с колонками:

- `text` — назначение/описание платежа (строка)
- `merchant` — мерчант/магазин (строка, можно пусто)
- `amount` — сумма (число)
- `category` — целевая категория (строка)

Пример датасета (синтетика) лежит здесь: `data/transactions_labeled.csv`.

Отдельная тестовая выборка для проверки качества: `data/transactions_test.csv`.

Неразмеченная выборка (без `category`) для “прогонов/ручной проверки”: `data/transactions_unlabeled.csv`.

## Обучение

### Через модульный CLI

```powershell
python -m expense_classifier.cli train --data .\data\transactions_labeled.csv --out .\model.joblib
```

### Через scripts (как в примерах)

```powershell
python .\scripts\train.py --data .\data\transactions_labeled.csv --out .\model.joblib
```

### Выбор модели

```powershell
# Logistic Regression (по умолчанию)
python -m expense_classifier.cli train --data .\data\transactions_labeled.csv --out .\model.joblib --model-type logreg

# Linear SVM + калибровка вероятностей
python -m expense_classifier.cli train --data .\data\transactions_labeled.csv --out .\model_svm.joblib --model-type svm
```

Примечание: на маленьком датасете метрики могут быть «шумными», это нормально для пет‑проекта.

## Классификация (predict)

### Текстовый вывод

```powershell
python -m expense_classifier.cli classify --model .\model.joblib --text "Корм для кота" --merchant "Petshop" --amount 890 --topk 3
```

Вывод примерно такой:

```text
category=pets confidence=0.47
	pets	0.47
	food	0.08
	transport	0.08
```

### JSON (для скриптов/интеграций)

```powershell
python -m expense_classifier.cli classify --model .\model.joblib --text "Корм для кота" --merchant "Petshop" --amount 890 --json --topk 3
```

Пример JSON:

```json
{"category":"pets","confidence":0.469,"topk":[{"category":"pets","confidence":0.469},{"category":"food","confidence":0.078},{"category":"transport","confidence":0.077}]}
```

## Идеи для улучшений (если захочется)

## Оценка качества на тестовой выборке

Если модель уже обучена и сохранена в `.joblib`, можно прогнать её на размеченном CSV и посмотреть метрики:

```powershell
python -m expense_classifier.cli evaluate --model .\model.joblib --data .\data\transactions_test.csv
```

## Прогон по неразмеченному CSV (predict-csv)

Если есть CSV без `category` (например `data/transactions_unlabeled.csv`), можно получить предсказания файлом:

```powershell
python -m expense_classifier.cli predict-csv --model .\model.joblib --data .\data\transactions_unlabeled.csv --out .\data\transactions_unlabeled_pred.csv --topk 3
```

В выходном CSV появятся колонки:

- `predicted_category`
- `confidence`
- (опционально) `top1_*`, `top2_*`, … если указан `--topk`

- добавить FastAPI mini‑API (`POST /classify`)
- расширить категории и датасет реальными (обезличенными) транзакциями
- добавить нормализацию текста (лейматизация/морфология) и/или ruBERT
