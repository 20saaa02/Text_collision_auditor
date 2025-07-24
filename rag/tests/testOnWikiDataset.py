import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from presentation.app import AnticollisionMainClass
from entity.dataApp import (
    prepareDBGet, checkCollisionOneGet, checkCollisionOneResult, ErrorClass
)


def processTests(text, contradict, neutral):

    service = AnticollisionMainClass()
    prepareDBRes = service.prepareDB(prepareDBGet(text=text))
    if prepareDBRes.error.isError:
        print(f"Ошибка при подготовке базы знаний: {prepareDBRes.error.messageError}")
        exit(1)

    checkCollisionOneResContradict = service.checkCollisionOne(checkCollisionOneGet(question=contradict))
    if checkCollisionOneResContradict.error.isError:
        print(f"Ошибка при поиске коллизий (contradict): {checkCollisionOneResContradict.error.messageError}")
        exit(2)
    ansContradict = len(checkCollisionOneResContradict.arrCollisions) != 0

    checkCollisionOneResNeutral = service.checkCollisionOne(checkCollisionOneGet(question=neutral))
    if checkCollisionOneResNeutral.error.isError:
        print(f"Ошибка при поиске коллизий (neutral): {checkCollisionOneResNeutral.error.messageError}")
        exit(2)
    #checkCollisionOneResNeutral = checkCollisionOneResult(list(), ErrorClass(False, ""))
    ansNeutral = len(checkCollisionOneResNeutral.arrCollisions) == 0

    return [ansContradict,
            ansNeutral,
            checkCollisionOneResContradict.arrCollisions,
            checkCollisionOneResNeutral.arrCollisions]


if __name__ == "__main__":
    df = pd.read_csv(r"test_wiki.csv")
    df['pred_contr'] = ["" for _ in range(len(df))]
    df['pred_neutr'] = ["" for _ in range(len(df))]
    df['col_contr'] = ["" for _ in range(len(df))]
    df['col_neutr'] = ["" for _ in range(len(df))]
    with open("res.txt", "w") as file:
        for index, row in df.iterrows():
            #if index < 4: continue
            if index > 3: break
            corpus = row['corpus']
            contradict = row['contradict']
            neutral = row['neutral']

            ansContradict, ansNeutral, arrColContradict, arrColNeutral = processTests(
                corpus, contradict, neutral)
            print(f'index {index}: {ansContradict}, {ansNeutral}')

            df.at[index, 'pred_contr'] = 1 if ansContradict else 0
            df.at[index, 'pred_neutr'] = 1 if ansNeutral else 0
            df.at[index, 'col_contr'] = arrColContradict if ansContradict else 0
            df.at[index, 'col_neutr'] = arrColNeutral if ansNeutral else 0

            file.write(f"{index}, {df.at[index, 'pred_contr']}, {df.at[index, 'pred_neutr']}\n")

    df.to_csv(
        r"test_wiki(YGPT).csv")