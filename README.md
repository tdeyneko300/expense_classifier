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
.\.venv\Scripts\Activate.ps1

# 2) Зависимости
pip install -r requirements.txt

# 3) Обучение (на синтетических данных из репозитория)
python .\scripts\train.py --data .\data\transactions_labeled.csv --out .\model.joblib

# 4) Предсказание категории
python -m expense_classifier.cli classify --model .\model.joblib --text "Yandex Go" --merchant "Yandex Go" --amount 420 --topk 3
