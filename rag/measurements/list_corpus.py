import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent # путь к папке rag
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from rag.presentation.app import AnticollisionMainClass
from rag.entity.dataApp import prepareDBGet, checkCollisionOneGet


if __name__ == "__main__":
    texts = pd.read_json(r'C:\Users\memel\PycharmProjects\Text_collision_auditor\DATA\list_corpus.jsonl', lines=True)
    queries = pd.read_json(r"C:\Users\memel\PycharmProjects\Text_collision_auditor\DATA\list_queries.jsonl", lines=True)
    text = texts.loc[texts.doc_id == 13497630].abstract.iloc[0]
    query = queries.loc[queries.id == 40].claim.iloc[0]
    
    united_text = text.append(query)





    service = AnticollisionMainClass()
    # загрузить текст в базу знаний
    prepareDBRes = service.prepareDB(prepareDBGet(text=text))
    if prepareDBRes.error.isError:
        print(prepareDBRes.error.messageError)
        exit(1)

    # проверить коллизию с вопросом
    checkCollisionOneRes = service.checkCollisionOne(checkCollisionOneGet(question=query))
    if checkCollisionOneRes.error.isError:
        print(checkCollisionOneRes.error.messageError)
        exit(2)
    print("Ответ модели (коллизии):")
    for i, ans in enumerate(checkCollisionOneRes.arrCollisions, 1):
        print(f"{i}: {ans}")


