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

Запуск

```bash
# Для просмотра справки запустить `poetry run tree --help`
poetry run tree
```

![image](https://user-images.githubusercontent.com/874234/167255789-f340808f-c69d-4269-b64b-ee2ca0d2c1a4.png)


### TASK 9 automatic hyperparameter with nested cross-validation

Запуск

```bash
# Для просмотра справки запустить `poetry run gridcv --help`
poetry run gridcv
```

![image](https://user-images.githubusercontent.com/874234/167406329-3df36202-07c4-4ef5-a8a3-09a558ad3550.png)

## Development

При инициализации проекта для разработки используйте команду

```bash
poetry install
```

Для тестов используйте [Pytest](https://docs.pytest.org)

```bash
poetry run pytest
```

![image](https://user-images.githubusercontent.com/874234/167427976-b52a6ea2-26bf-44e3-9773-65d1313a509e.png)

Для форматирования используйте [Black](https://black.readthedocs.io)

```bash
black src/eval_sel/

# All done! ✨ 🍰 ✨
# 6 files left unchanged.
```

![image](https://user-images.githubusercontent.com/874234/167435179-784dd23d-4bfd-4210-9cc0-8d15ef2f6114.png)

Проверка форматирования организована через [flake8](https://flake8.pycqa.org/en/latest/#)
Используйте следующую команду для запуска этой проверки
```bash
flake8 src/eval_sel/
```

Для проверки static typing используйте [mypy](http://www.mypy-lang.org)

```bash
mypy src/eval_sel/
```

![image](https://user-images.githubusercontent.com/874234/167434683-6364b7f3-944b-4201-8a10-8bc2b959937f.png)

### Nox

Запуск всех команд доступен через [Nox](https://nox.thea.codes/en/stable/)

```bash
nox
```

![image](https://user-images.githubusercontent.com/874234/167442459-0a49a69b-58db-4e6f-8a6c-c87a97067bc4.png)
