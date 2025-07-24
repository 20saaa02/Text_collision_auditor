import sys
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

# --- Конфигурация ---
MAX_WORKERS = 8
ROWS_TO_PROCESS = 3

# --- Настройка путей ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from presentation.app import AnticollisionMainClass
from entity.dataApp import prepareDBGet, checkCollisionOneGet
# ИЗМЕНЕНИЕ: Импортируем саму модель, чтобы загрузить ее один раз
from sentence_transformers import SentenceTransformer


# ИЗМЕНЕНИЕ: Функция теперь принимает готовую модель
def processTests(text, contradict, neutral, embedding_model):
    # Каждый поток создает свой сервис, но использует ОБЩУЮ модель
    service = AnticollisionMainClass(embedding_model=embedding_model)

    # ВАЖНО: Убираем из getData параметр loadDBbtw, т.к. его нет в определении класса
    prepareDBRes = service.prepareDB(prepareDBGet(text=text, loadDBbtw=False))
    if prepareDBRes.error.isError:
        return [False, False, [f"Ошибка prepareDB: {prepareDBRes.error.messageError}"], []]

    checkCollisionOneResContradict = service.checkCollisionOne(checkCollisionOneGet(question=contradict))
    if checkCollisionOneResContradict.error.isError:
        print(f"Ошибка при поиске коллизий (contradict): {checkCollisionOneResContradict.error.messageError}")
    ansContradict = len(checkCollisionOneResContradict.arrCollisions) != 0

    checkCollisionOneResNeutral = service.checkCollisionOne(checkCollisionOneGet(question=neutral))
    if checkCollisionOneResNeutral.error.isError:
        print(f"Ошибка при поиске коллизий (neutral): {checkCollisionOneResNeutral.error.messageError}")
    ansNeutral = len(checkCollisionOneResNeutral.arrCollisions) == 0

    return [ansContradict,
            ansNeutral,
            checkCollisionOneResContradict.arrCollisions,
            checkCollisionOneResNeutral.arrCollisions]


# ИЗМЕНЕНИЕ: Worker теперь тоже принимает готовую модель
def process_row_worker(args):
    """
    Принимает кортеж (индекс, данные строки, общая_модель), обрабатывает его
    и возвращает кортеж (индекс, результат).
    """
    index, row, shared_model = args
    corpus = row['corpus']
    contradict = row['contradict']
    neutral = row['neutral']

    result_data = processTests(corpus, contradict, neutral, shared_model)

    return index, result_data


if __name__ == "__main__":
    # start_time = time.time()
    # --- ШАГ 1: ЗАГРУЖАЕМ ТЯЖЕЛУЮ МОДЕЛЬ ОДИН РАЗ ---
    # print("Загрузка модели SentenceTransfqormer... (это может занять время)")
    shared_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    start_time = time.time()
    # print("Модель успешно загружена и готова к использованию.")

    df = pd.read_csv(r"test_wiki.csv")

    if ROWS_TO_PROCESS is not None:
        df = df.head(ROWS_TO_PROCESS)

    # --- ШАГ 2: ПОДГОТАВЛИВАЕМ ЗАДАЧИ, ВКЛЮЧАЯ ССЫЛКУ НА ОБЩУЮ МОДЕЛЬ ---
    tasks = [(index, row, shared_embedding_model) for index, row in df.iterrows()]

    # print(f"Запускаем параллельную обработку {len(tasks)} строк в {MAX_WORKERS} потоков...")

    # --- ШАГ 3: ЗАПУСКАЕМ ПУЛ ПОТОКОВ ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_iterator = executor.map(process_row_worker, tasks)

        for index, result_data in tqdm(results_iterator, total=len(tasks), desc="Обработка строк"):
            ansContradict, ansNeutral, arrColContradict, arrColNeutral = result_data

            df.at[index, 'pred_contr'] = 1 if ansContradict else 0
            df.at[index, 'pred_neutr'] = 1 if ansNeutral else 0
            # Сохраняем как строку, т.к. CSV не умеет хранить списки напрямую
            # df.at[index, 'col_contr'] = str(arrColContradict)
            # df.at[index, 'col_neutr'] = str(arrColNeutral)

    # print("Обработка завершена. Сохранение результатов...")

    output_filename = r"test_wiki(YGPT, new prompt with NO).csv"
    df.to_csv(output_filename, index=False)

    # print(f"Результаты сохранены в файл: {output_filename}")

    print(f"Время выполнения: {time.time() - start_time}")
