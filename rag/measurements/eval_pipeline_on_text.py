import sys
from pathlib import Path
import pandas as pd

# --- Настройка путей (как в вашем коде) ---
# Убедитесь, что этот путь правильный. Если запускаете из папки, где лежит скрипт,
# а папка rag на уровень выше, используйте Path(__file__).parent.parent
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Импорт ваших классов и сущностей ---
from presentation.app import AnticollisionMainClass
from entity.dataApp import (
    factsIntoEmbeddingsGet,
)
from entity.dataLLM import findTopNearestLLMGet, findCollisionsGet
from entity.dataDB import findTopNearestDBGet, loadEmbeddingsDBGet

# --- Константы (взяты из вашего кода) ---
COUNT_NEAR_FIND_DB = 30
COUNT_NEAR_FIND_LLM = 5  # Количество для реранкера
NOF_NEAREST_CELLS_TO_CHECK = 10

if __name__ == "__main__":
    # 1. Загрузка и подготовка данных
    texts_df = pd.read_json(r'C:\Users\memel\PycharmProjects\Text_collision_auditor\DATA\list_corpus.jsonl', lines=True)
    queries_df = pd.read_json(r"C:\Users\memel\PycharmProjects\Text_collision_auditor\DATA\list_queries.jsonl",
                              lines=True)

    text_content = texts_df.loc[texts_df.doc_id == 13497630, 'abstract'].iloc[0]
    query_content = queries_df.loc[queries_df.id == 40, 'claim'].iloc[0]
    #print(text_content)
    #print(query_content)
    text_content.append(query_content)
    sentences = text_content
    #print(sentences)

    # 2. Инициализация сервиса
    service = AnticollisionMainClass()

    # 3. Подготовка векторной базы (RAG)
    # Так как ваш public метод prepareDB не предназначен для списка,
    # мы аккуратно вызываем его внутренние компоненты для корректной работы.

    # 3.1. Превращаем предложения в эмбеддинги
    # Используем внутренний метод __factsIntoEmbeddings через "name mangling"
    facts_res = service.__factsIntoEmbeddings(factsIntoEmbeddingsGet(sentences=sentences))
    if facts_res.error.isError:
        print(f"Ошибка при создании эмбеддингов: {facts_res.error.messageError}")
        exit(1)
    sentence_embeddings = facts_res.embeddings

    # 3.2. Загружаем эмбеддинги и предложения в RAG
    load_res = service.RAGRetriever.loadEmbeddingsDB(
        loadEmbeddingsDBGet(sentence_embeddings=sentence_embeddings, sentences=sentences)
    )
    if load_res.error.isError:
        print(f"Ошибка при загрузке в RAG: {load_res.error.messageError}")
        exit(1)

    # 4. Основной цикл проверки на коллизии
    total_collisions_found = 0
    for i, query_sentence in enumerate(sentences):
        # if i < 13:
        #     continue
        query_embedding = sentence_embeddings[i]

        # Шаг 1: Поиск топ-K похожих в RAG. Ищем K+1, чтобы потом отфильтровать сам запрос.
        find_db_res = service.RAGRetriever.findTopNearestDB(
            findTopNearestDBGet(
                query_embedding=query_embedding,
                k=COUNT_NEAR_FIND_DB + 1,
                NofNearestCellsToCheck=NOF_NEAREST_CELLS_TO_CHECK
            )
        )
        if find_db_res.error.isError:
            print(f"Ошибка поиска в RAG для предложения #{i}: {find_db_res.error.messageError}")
            continue

        # Отфильтровываем сам исходный запрос из кандидатов
        candidate_sentences = find_db_res.topNearest[1:]

        if not candidate_sentences:
            continue

        # Шаг 2: Реранкер (LLM) для выбора самых релевантных кандидатов
        rerank_res = service.LLMService.findTopNearestLLM(
            findTopNearestLLMGet(
                question=query_sentence,
                topFacts=candidate_sentences,
                countFind=COUNT_NEAR_FIND_LLM
            )
        )
        if rerank_res.error.isError:
            print(f"Ошибка реранкинга для предложения #{i}: {rerank_res.error.messageError}")
            continue

        reranked_sentences = rerank_res.topNearest
        if not reranked_sentences:
            continue

        # Шаг 3: Финальный поиск коллизий среди лучших кандидатов
        collisions_res = service.LLMService.findCollisions(
            findCollisionsGet(
                question=query_sentence,
                topFacts=reranked_sentences
            )
        )
        if collisions_res.error.isError:
            print(f"Ошибка поиска коллизий для предложения #{i}: {collisions_res.error.messageError}")
            continue

        # Шаг 4: Вывод результатов в заданном формате
        if collisions_res.arrCollisionResult:
            total_collisions_found += len(collisions_res.arrCollisionResult)
            print(f"i: {i}")
            print(f"query: {query_sentence}")
            print("collisions:")
            for num, collision_text in enumerate(collisions_res.arrCollisionResult, 1):
                print(f"    {num}) {collision_text}")
            print("-" * 20)  # Разделитель для наглядности

    if total_collisions_found == 0:
        print("\nАнализ завершен. Внутренних противоречий не обнаружено.")
    else:
        print(f"\nАнализ завершен. Всего найдено коллизий: {total_collisions_found}.")