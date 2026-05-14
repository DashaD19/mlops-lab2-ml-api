![CI](https://github.com/DashaD19/mlops-lab2-ml-api/actions/workflows/ci.yml/badge.svg)

# Wine ML API — лабораторна робота № 2 «CI/CD та ML API»

Курс «MLOps у веб-системах», магістратура КПІ, 10 семестр. Варіант 2.

## Опис проєкту

Проєкт реалізує наскрізний MLOps-конвеєр для задачі багатокласової класифікації на датасеті Wine зі складу `scikit-learn`. Навчена модель SVC у Pipeline зі StandardScaler виставлена через REST API на FastAPI, упакована у Docker-образ та автоматично перевіряється GitHub Actions при кожному `push` і `pull request`. Методичні вказівки наводять приклад на класифікаторі Iris із LogisticRegression; у цьому варіанті використано Wine + SVC — найкращу конфігурацію SVC (`svc_rbf_C1`: ядро RBF, C=1.0, accuracy 0.9722) з серії з семи експериментів лабораторної роботи № 1 ([`mlops-lab1-wine`](https://github.com/DashaD19/mlops-lab1-wine)), де через MLflow tracking порівнювалися SVC, LogisticRegression та GradientBoosting. Кінцева мета — отримати публічно доступний endpoint інференсу, розгорнутий на безкоштовному тарифі Render, та продемонструвати, що кожна зміна коду спочатку перевіряється автоматизованими тестами, а лише потім потрапляє у продакшен-середовище.

Сервіс експонує три ендпоінти: `GET /` повертає статус, `GET /health` використовується як liveness probe для платформи розгортання, а `POST /predict` приймає 13 фізико-хімічних характеристик зразка вина та повертає прогнозований клас разом з оцінкою ймовірності.

## Стек технологій

- **Python 3.11** — основна мова реалізації, фіксована у `Dockerfile` та GitHub Actions.
- **FastAPI 0.115** + **Uvicorn 0.30** — асинхронний веб-фреймворк та ASGI-сервер для production-запуску.
- **Pydantic 2.9** — декларативна валідація вхідних JSON-тіл; помилковий тип автоматично перетворюється на `422 Unprocessable Entity`.
- **scikit-learn 1.5** + **joblib 1.4** — тренування `Pipeline(StandardScaler → SVC)` та серіалізація моделі у файл `model.joblib`.
- **pytest 8.3** + **httpx 0.27** — фреймворк тестування та HTTP-клієнт для `TestClient` FastAPI.
- **Docker** (базовий образ `python:3.11-slim`) — контейнеризація для відтворюваного запуску.
- **GitHub Actions** — CI workflow з двома задачами: `test` (юніт-тести + тренування) та `docker-build` (перевірка коректності `Dockerfile`).
- **Render** — PaaS-платформа для публічного розгортання Docker-образу з безкоштовним тарифом.

## Як запустити локально

Передумови: встановлений Python 3.11 та `git`.

```bash
git clone https://github.com/DashaD19/mlops-lab2-ml-api.git
cd mlops-lab2-ml-api

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

pip install --upgrade pip
pip install -r requirements.txt

python -m ml.train                 # створює model.joblib у корені проєкту
uvicorn app.main:app --reload      # старт API на http://localhost:8000
```

Після старту в браузері за адресою [http://localhost:8000/docs](http://localhost:8000/docs) доступна інтерактивна документація Swagger UI. У ній можна розгорнути блок `POST /predict`, натиснути «Try it out» та надіслати тестовий запит безпосередньо з браузера — це найшвидший спосіб переконатися, що інференс працює коректно.

## Запуск через Docker

Інструкції нижче зберуть образ та запустять контейнер на порту 8000. Модель тренується під час самої збірки, тому артефакт `model.joblib` потрапляє всередину образу й контейнер стартує без додаткових кроків.

```bash
docker build -t ml-api:lab2 .
docker run --rm -p 8000:8000 ml-api:lab2
```

Перевірити, що сервіс відповідає всередині контейнера:

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true}
```

## Як запустити тести

```bash
pytest -q
```

Очікуваний вивід — шість зелених тестів. Набір складається з двох рівнів:

- **`tests/test_model.py`** — перевіряє, що скрипт `ml/train.py` створює файл моделі у тимчасовому каталозі `tmp_path`, повертає `accuracy` у валідному діапазоні `[0.0, 1.0]` та що передбачення на типовому зразку класу_0 є одним із трьох допустимих ідентифікаторів.
- **`tests/test_api.py`** — піднімає застосунок через `fastapi.testclient.TestClient` та перевіряє відповіді ендпоінтів `/`, `/health`, `/predict` (валідний payload із 13 ознак) і `/predict` з рядком замість числа (Pydantic має повернути HTTP 422).

Перед запуском тестів API файл моделі створюється автоматично, якщо його ще немає у корені проєкту — це гарантує, що CI-середовище зі склонованим «начисто» репозиторієм пройде успішно.

## Як працює API

Ендпоінт **`GET /`** повертає коротку службову інформацію та підтверджує, що ASGI-процес запущено:

```bash
curl http://localhost:8000/
# {"status":"ok","service":"Wine ML API"}
```

Ендпоінт **`GET /health`** використовується Render для liveness-перевірки та повідомляє, чи завантажена модель у пам'ять процесу:

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true}
```

Ендпоінт **`POST /predict`** приймає JSON з тринадцятьма числовими полями (порядок збігається з `sklearn.datasets.load_wine().feature_names`):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127.0,
    "total_phenols": 2.8,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280_od315_of_diluted_wines": 3.92,
    "proline": 1065.0
  }'
# {"class_id":0,"class_name":"class_0","probability":0.9878}
```

Якщо тип будь-якого поля невалідний (наприклад, рядок замість числа), Pydantic поверне `422 Unprocessable Entity` без виконання інференсу:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol":"not-a-number"}'
# HTTP/1.1 422 Unprocessable Entity
```

Модель завантажується одноразово у хуку `@app.on_event("startup")` і тримається у глобальній змінній — це уникає десятків мілісекунд накладних витрат на повторне читання `model.joblib` з диска при кожному запиті.

## Посилання на деплой

Сервіс розгорнуто на Render у середовищі Docker (free tier):

- **Публічний URL:** https://mlops-lab2-ml-api.onrender.com
- **Liveness:** https://mlops-lab2-ml-api.onrender.com/health
- **Swagger UI:** https://mlops-lab2-ml-api.onrender.com/docs

> Безкоштовний тариф Render «засинає» після приблизно 15 хвилин відсутності трафіку, тому перший запит після паузи може зайняти 30–60 секунд (cold start). Подальші запити обробляються миттєво.

## Структура репозиторію

```
mlops-lab2-ml-api/
├── .github/
│   └── workflows/
│        └── ci.yml                  # GitHub Actions workflow
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI застосунок
│   └── schemas.py                   # Pydantic-моделі вхід/вихід
├── ml/
│   ├── __init__.py
│   └── train.py                     # Скрипт тренування
├── tests/
│   ├── __init__.py
│   ├── test_api.py                  # Тести API
│   └── test_model.py                # Тести моделі
├── model.joblib                     # Артефакт навченої моделі (створюється train.py)
├── requirements.txt                 # Залежності Python
├── Dockerfile                       # Інструкції для Docker-образу
├── .dockerignore
└── README.md
```
