# TITANIC: MACHINE LEARNING FROM DISASTER

Бинарная классификация: предсказание выживания пассажиров «Титаника».

## Установка

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
pyyaml
lightgbm
catboost
torch
```
## Данные

```
Датасет нужно скачать отдельно с Kaggle - не пушил CSV:
[Titanic — Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

Положи файлы в папку `data/` под именами:
- `data/train_titanic.csv`
- `data/test_titanic.csv`
```

## Запуск

```bash
python main.py --model dnn
```

Результат сохраняется в `submissions/submission_titanic.csv`.

## Пайплайн

```
load_data (raw csv)
    ↓
build_titanic_features  (FE: Title, FamilySize, IsAlone, AgeGroup, FarePerPerson, HasCabin)
    ↓
split_train_val  (80/20, до препроцессинга - без утечки)
    ↓
TabularPreprocessor:
    • drop_columns (PassengerId, Name, Ticket, Cabin)
    • fillna: Age => median, Fare => median, Embarked => mode (fit на train)
    • ordinal: AgeGroup  =>  {Kid:0, Teen:1, Adult:2, Old:3}
    • one-hot: Sex, Embarked, Title (fit на train, transform на val/test)
    • StandardScaler: Age, Fare, FarePerPerson
    ↓
train_dnn_model  (DNN + CV, early stopping)
    ↓
make_submission
```

## Результаты

| Модель                | CV Accuracy | Holdout Accuracy | Kaggle Score |
|-----------------------|-------------|------------------|--------------|
| Dummy (most_frequent) | 0.6145      | -                | -            |
| Logistic Regression   | 0.8217      | -                | -            |
| Random Forest         | 0.8259      | -                | -            |
| LightGBM              | 0.8146      | -                | -            |
| CatBoost              | 0.8245      | -                | -            |
| **MLP [64, 32]** 👑   | **0.8238**  | **0.8212**       | **0.78468**  |

**Финальная модель:** MLP [64, 32] (PyTorch, `BCEWithLogitsLoss`, early stopping).

## Структура проекта

```text
titanic-ml/
├── 📁 configs/
│   └── 📄 titanic.yaml             # YAML-конфиг проекта
│
├── 📁 data/                        # Сырые данные (в .gitignore)
│   ├── 📄 train_titanic.csv
│   └── 📄 test_titanic.csv
│
├── 📁 notebooks/
│   ├── 📓 EDA_Titanic.ipynb
│   └── 📓 Modeling_Titanic.ipynb
│
├── 📁 src/                         # Модули пайплайна
│   ├── 🐍 __init__.py
│   ├── 🐍 config.py                # Загрузка YAML-конфигов
│   ├── 🐍 data.py                  # Загрузка данных и сплит
│   ├── 🐍 features.py              # FE + TabularPreprocessor
│   ├── 🐍 models.py                # Фабрика моделей + MLP (PyTorch)
│   ├── 🐍 train.py                 # CV, обучение моделей и DNN
│   ├── 🐍 predict.py               # Генерация submission
│   └── 🐍 utils.py                 # Метрики, сиды, утилиты
│
├── 📁 submissions/                 # Результаты (в .gitignore, оставлен .gitkeep)
│   └── 📄 .gitkeep
│
├── 🐍 main.py
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md
```

## Ключевые принципы

- **Препроцессор фитится только на train.** В val/test применяется `transform`,
  статистики (медианы, моды, скейлер, OHE-категории) не пересчитываются.
  Утечек нет.
- **One-hot fit на train, transform на val/test.** Новые категории кодируются
  как «все нули» (`handle_unknown="ignore"`) - без рассинхрона форм данных.
- **Early stopping + возврат к лучшему чекпоинту** в DNN (`train_dnn` в `src/models.py`).
- **Единый источник истины:** гиперпараметры и пути - только в `configs/titanic.yaml`.
  `src/config.py` только загружает и разрешает пути.

## Ноутбуки

### EDA
- **`EDA_Titanic.ipynb`** - пропуски, распределения, корреляции,
  выводы о значимых признаках (Sex, Pclass, Fare, Title).

### Моделирование
- **`Modeling_Titanic.ipynb`** - сравнение моделей через 5-fold CV,
  MLP [64, 32] обучена через честную CV с `BCEWithLogitsLoss`.

## Тюнинг гиперпараметров

Секция `tuning` в `configs/titanic.yaml` - заготовка под Optuna - **так и не дошли до нее руки**
Сейчас `enabled: false`: гиперпараметры подобраны вручную в ноутбуке моделирования.
