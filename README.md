# Model evaluation and selection

Приложение с моделью
для [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction/overview), созданная в
рамках курса от [Rolling Scopes School](https://github.com/rolling-scopes-school). Идеи для шаблона проекта взяты
из [python-packaging](https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure)
и [ml-project-demo](https://github.com/rolling-scopes-school/ml-project-demo)

При разработке использовался:

- [Python](https://www.python.org/downloads/release/python-3912/) version 3.9.12 (под Windows 11)
- [Poetry](https://python-poetry.org/) version 1.1.13

## Установка приложения

1. Запустить команду ```poetry install --no-dev``` для установки всех зависимостей, которые перечислены в
   файле `pyproject.toml`
2. Загрузить файлы [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) в
   директорию проекта `data`

## Использование приложения

1. Просмотр доступных параметров приложения и их значений по умолчанию

```
poetry run train --help
```

2. Запуск приложения с параметрами по умолчанию

```
poetry run train
```

Будет загружены информация из файла `data/train.csv` и результаты запуска будут сохранены в директории `models` и `mlruns`

3. Просмотр результатов через [MLflow](https://www.mlflow.org)

```
poetry run mlflow ui
```

### TASK 8

![image](https://user-images.githubusercontent.com/874234/167255789-f340808f-c69d-4269-b64b-ee2ca0d2c1a4.png)