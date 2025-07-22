import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent # путь к папке rag
sys.path.append(str(project_root))


from sentence_transformers import SentenceTransformer

from service.LLMService import LLMService
from service.preprocessingDataService import PreprocessingDataService
from service.RAGRetriever import RAGRetrieverGlobal

from entity.dataPreproc import splittingTextIntoChunksGet, cleanTextGet, initPreprocessingDataServiceGet
from entity.dataLLM import splittingChunksIntoFactsGet, initLLMServiceGet, findTopNearestLLMGet, findCollisionsGet
from entity.dataApp import textIntoFactsGet, textIntoFactsResult, ErrorClass, prepareDBResult, prepareDBGet,\
    factsIntoEmbeddingsGet, factsIntoEmbeddingsResult, checkCollisionOneGet, checkCollisionOneResult
from entity.dataDB import findTopNearestDBGet, loadEmbeddingsDBGet, initDBGet


SAVE_DB_FILE_PATH = "../file/database/save_db.txt"
COUNT_NEAR_FIND_DB = 30
COUNT_NUAR_FIND_LLM = 5
NOF_NEAREST_CELLS_TO_CHECK = 10 # потестить

# Основной класс запуска RAG
class AnticollisionMainClass:
    def __init__(self):
        self.LLMService = LLMService(initLLMServiceGet())
        self.PreprocessingDataService = PreprocessingDataService(initPreprocessingDataServiceGet())
        self.RAGRetrieverGlobal = RAGRetrieverGlobal(initDBGet(db_path=SAVE_DB_FILE_PATH))
        # Инициализация модели для эмбеддингов
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Легкая модель
        # Или для русского языка: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

    # ПОДГОТОВКА БАЗЫ ЗНАНИЙ
    def prepareDB(self, getData: prepareDBGet) -> prepareDBResult:
        prepareDBRes = prepareDBResult(error=ErrorClass(False, ""))
        # ПРЕДОБРАБОТКА ТЕКСТА
        textIntoFactsRes = self.__textIntoFacts(textIntoFactsGet(text=getData.text))
        if textIntoFactsRes.error.isError:
            prepareDBRes.error = textIntoFactsRes.error
            return prepareDBRes
        # ФАКТЫ В ЭМБЕДДИНГИ
        factsIntoEmbeddingsRes = self.__factsIntoEmbeddings(factsIntoEmbeddingsGet(textIntoFactsRes.facts))
        if factsIntoEmbeddingsRes.error.isError:
            prepareDBRes.error = factsIntoEmbeddingsRes.error
            return prepareDBRes
        # ЭМБЕДДИНГИ ЗАПИСЫВАЕМ В БАЗУ
        loadEmbeddingsDBRes = self.RAGRetrieverGlobal.loadEmbeddingsDB(loadEmbeddingsDBGet(sentence_embeddings=factsIntoEmbeddingsRes.embeddings, sentences=textIntoFactsRes.facts))
        if loadEmbeddingsDBRes.error.isError:
            prepareDBRes.error = loadEmbeddingsDBRes.error
        return prepareDBRes


    # ПРОВЕРКА КОЛЛИЗИИ ВОПРОСА С ФАКТАМИ ИЗ БАЗЫ ЗНАНИЙ
    def checkCollisionOne(self, getData: checkCollisionOneGet) -> checkCollisionOneResult:
        checkCollisionOneRes = checkCollisionOneResult(list(), ErrorClass(False, ""))
        # ВОПРОС В ЭМБЕДДИНГ
        factsIntoEmbeddingsRes = self.__factsIntoEmbeddings(factsIntoEmbeddingsGet(list(getData.question)))
        if factsIntoEmbeddingsRes.error.isError:
            checkCollisionOneRes.error = factsIntoEmbeddingsRes.error
            return checkCollisionOneRes
        # ПОИСК 30 БЛИЖАЙШИХ ПО БАЗЕ ЗНАНИЙ
        findTopNearestDBRes = self.RAGRetrieverGlobal.findTopNearestDB(findTopNearestDBGet(query_embedding=factsIntoEmbeddingsRes.embeddings, k=COUNT_NEAR_FIND_DB,\
                                                                                            NofNearestCellsToCheck=NOF_NEAREST_CELLS_TO_CHECK))
        if findTopNearestDBRes.error.isError:
            checkCollisionOneRes.error = findTopNearestDBRes.error
            return checkCollisionOneRes
        # ПОИСК 5 БЛИЖАЙШИХ ИЗ 30 ЧЕРЕЗ ЛЛМ
        findTopNearestLLMRes = self.LLMService.findTopNearestLLM(findTopNearestLLMGet(question=getData.question, topFacts=findTopNearestDBRes.topNearest, countFind=COUNT_NUAR_FIND_LLM))
        if findTopNearestLLMRes.error.isError:
            checkCollisionOneRes.error = findTopNearestLLMRes.error
            return checkCollisionOneRes
        # ЛЛМ ИЩЕТ КОЛЛИЗИИ
        findCollisionsRes = self.LLMService.findCollisions(findCollisionsGet(question=getData.question, topFacts=findTopNearestLLMRes.topNearest))
        if findCollisionsRes.error.isError:
            checkCollisionOneRes.error = findCollisionsRes.error
            return checkCollisionOneRes
        checkCollisionOneRes.arrCollisions = findCollisionsRes.arrCollisionResult
        return checkCollisionOneRes


    # ПРЕДОБРАБОТКА ТЕКСТА
    def __textIntoFacts(self, getData: textIntoFactsGet) -> textIntoFactsResult:
        textIntoFactsRes = textIntoFactsResult(error=ErrorClass(False, ""), facts=list())
        print("Полученный текст:", getData.text)

        # разбивка на чанки
        splittingTextIntoChunksRes = self.PreprocessingDataService.splittingTextIntoChunks(splittingTextIntoChunksGet(text=getData.text))
        if splittingTextIntoChunksRes.error.isError:
            textIntoFactsRes.error = splittingTextIntoChunksRes.error
            return textIntoFactsRes
        chunks = splittingTextIntoChunksRes.chunks
        print("Полученные чанки:", chunks)

        # очистка текста (чанков по отдельности)
        for i in range(len(chunks)):
            cleanTextRes = self.PreprocessingDataService.cleanText(cleanTextGet(text = chunks[i]))
            chunks[i] = cleanTextRes.text
        print("Очищенные чанки:", chunks)

        # выделение из чанков фактов
        for i in range(len(chunks)):
            splittingChunksIntoFactsRes = self.LLMService.splittingChunksIntoFacts(splittingChunksIntoFactsGet(chunk=chunks[i]))
            if splittingChunksIntoFactsRes.error.isError:
                textIntoFactsRes.error = splittingChunksIntoFactsRes.error
                return textIntoFactsRes
            textIntoFactsRes.facts += splittingChunksIntoFactsRes.arrFacts
        print("Полученные факты:", textIntoFactsRes.facts)

        return textIntoFactsRes
    

    # Преобразует список предложений в массив эмбеддингов
    def __factsIntoEmbeddings(self, getData: factsIntoEmbeddingsGet) -> factsIntoEmbeddingsResult:
        factsIntoEmbeddingsRes = factsIntoEmbeddingsResult(None, ErrorClass(False, ""))
        try:
            factsIntoEmbeddingsRes.embeddings = self.embedding_model.encode(getData.sentences)
        except Exception as e:
            factsIntoEmbeddingsRes.error = ErrorClass(True, f"Ошибка генерации эмбеддингов: {str(e)}")
        return factsIntoEmbeddingsRes






if __name__ == "__main__":
    # инициализация
    service = AnticollisionMainClass()

    #import os
    #print("Текущая рабочая директория:", os.getcwd())

    # получить текст и вопрос для проверки
    text = open("../file/test_collision_text.txt", "r").read()
    question = open("../file/test_collision_question.txt", "r").read()

    # загрузить текст в базу знаний
    prepareDBRes = service.prepareDB(prepareDBGet(text=text))
    if prepareDBRes.error.isError:
        print(prepareDBRes.error.messageError)
        exit(1)

    # проверить коллизию с вопросом
    checkCollisionOneRes = service.checkCollisionOne(checkCollisionOneGet(question=question))
    if checkCollisionOneRes.error.isError:
        print(checkCollisionOneRes.error.messageError)
        exit(2)
    print("Ответ модели (коллизии):")
    for i, ans in enumerate(checkCollisionOneRes.arrCollisions, 1):
        print(f"{i}: {ans}")