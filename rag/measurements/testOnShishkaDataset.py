import sys
from pathlib import Path
import pandas as pd
import ast

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from presentation.app import AnticollisionMainClass
from entity.dataApp import (
    prepareDBGet, checkCollisionOneGet
)


def processTests(sentences):

    service = AnticollisionMainClass()
    prepareDBRes = service.prepareDB(prepareDBGet(text=sentences))
    if prepareDBRes.error.isError:
        print(f"Ошибка при подготовке базы знаний: {prepareDBRes.error.messageError}")
        exit(1)

    collisions_res_with_last = set()
    for i, query_sentence in enumerate(sentences):
        if i < len(sentences) - 1:
            continue

        checkCollisionOneRes = service.checkCollisionOne(checkCollisionOneGet(question=sentences[i]))
        if checkCollisionOneRes.error.isError:
            print(f"Ошибка при поиске коллизий: {checkCollisionOneRes.error.messageError}")
            exit(2)

        if sentences[-1] in checkCollisionOneRes.arrCollisions:
            collisions_res_with_last.add(i)
        if i == (len(sentences) - 1):
            for collis in checkCollisionOneRes.arrCollisions:
                if collis in sentences:
                    collisions_res_with_last.add(sentences.index(collis))

    return collisions_res_with_last


if __name__ == "__main__":
    data_frame = pd.read_csv(r"C:\Users\memel\PycharmProjects\Text_collision_auditor1\rag\measurements\data_test_for_measure.csv")
    data_frame['pred_pair'] = [[] for _ in range(len(data_frame))]
    all_count_collis = 0
    all_count_collis_tests = 0
    find_count_collis_tests = 0
    not_find_count_collis = 0
    for index, row in data_frame.iterrows():
        #if index < 33:
        #    continue
        #if index > 100:
        #    break
        print(index, end=" ")

        text = ast.literal_eval(row['text'])

        # Преобразуем строку в список кортежей
        true_pairs = ast.literal_eval(row['true_pair'])
        right_results = {sent for (_, sent) in true_pairs}  # Берём только вторые элементы

        test_result = processTests(sentences=text)
        not_find_count_collis += len(right_results - test_result)
        all_count_collis += len(right_results)

        if len(right_results) > 0:
            all_count_collis_tests += 1
            if len(test_result) > 0:
                find_count_collis_tests += 1

        if test_result:
            data_frame.at[index, 'pred_pair'] = [
                (row['len_text']-1, sent) for sent in test_result
            ]

    find_count_collis = all_count_collis - not_find_count_collis
    print(f"Нашли коллизий {find_count_collis}/{all_count_collis}. Коллизионные тесты найдены {find_count_collis_tests}/{all_count_collis_tests}.")
    data_frame.to_csv(r"C:\Users\memel\PycharmProjects\Text_collision_auditor1\rag\measurements\data_test_for_measure(YGPT).csv",
                      index=False)
