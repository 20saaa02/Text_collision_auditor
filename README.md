# Text_collision_auditor

Сервис для нахождения коллизий в тексте

## Установка

Требования:

- Python 3.10

Создаем виртуальное окружение

```bash
python3 -m venv venv
```

Переходим в него

```bash
source venv/bin/activate
```

Устанавливаем зависимости

```bash
pip install -r requirements.txt
```

Запускаем сервис (rag)

```bash
python3 ./rag/app.py
```

# Правила разработки

- Модули и файлы — `snake_case`.
- Классы — `PascalCase`.
- Константы — `UPPER_SNAKE_CASE`.
- RAG пишем в папке rag, transformer аналогично
- Каждую фичу пишем в своей ветке (пример: `feature/rag_llm`).
- Название ветки/фичи начинается с rag или transformer.
- Текст коммита на английском
- Название коммита: для фичи — `feat: <название>`, для багфикса — `fix: <название>`.
- Комменты: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
- Сервисы возвращают и получают dataclasses, те датаклассы, которые сервис получает называется <название_класса>Get, а возвращает <название_класса>Result
- Все dataclasses записываются в файл data.py