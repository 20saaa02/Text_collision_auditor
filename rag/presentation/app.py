# presentation/app.py
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from service.LLMService import LLMService
from service.preprocessingDataService import PreprocessingDataService
from service.RAGRetriever import BaseRAGRetriever, RAGRetrieverGlobal

from entity.dataPreproc import splittingTextIntoChunksGet, cleanTextGet, initPreprocessingDataServiceGet
from entity.dataLLM import splittingChunksIntoFactsGet, initLLMServiceGet, findTopNearestLLMGet, findCollisionsGet
from entity.dataApp import textIntoFactsGet, textIntoFactsResult, ErrorClass, prepareDBResult, prepareDBGet, \
    factsIntoEmbeddingsGet, factsIntoEmbeddingsResult, checkCollisionOneGet, checkCollisionOneResult
from entity.dataDB import findTopNearestDBGet, loadEmbeddingsDBGet, initDBGet

SAVE_DB_FILE_PATH = "../file/database/save_db.txt"
COUNT_NEAR_FIND_DB = 30
COUNT_NUAR_FIND_LLM = 5
NOF_NEAREST_CELLS_TO_CHECK = 10


# Основной класс запуска RAG
class AnticollisionMainClass:
    # ИЗМЕНЕНИЕ: Принимаем готовую embedding_model как аргумент
    def __init__(self, embedding_model, is_outer: bool = False):
        self.RAGRetriever = None
        self.LLMService = LLMService(initLLMServiceGet())
        self.PreprocessingDataService = PreprocessingDataService(initPreprocessingDataServiceGet())
        self.is_outer = is_outer
        # ИЗМЕНЕНИЕ: Используем переданную модель, а не создаем новую
        self.embedding_model = embedding_model
        self.isLoadDB = True
        self.factsForFindCollisions = list()

    # ПОДГОТОВКА БАЗЫ ЗНАНИЙ
    def prepareDB(self, getData: prepareDBGet) -> prepareDBResult:
        prepareDBRes = prepareDBResult(error=ErrorClass(False, ""))

        # ПРЕДОБРАБОТКА ТЕКСТА
        textIntoFactsRes = self.__textIntoFacts(textIntoFactsGet(text=getData.text))
        if textIntoFactsRes.error.isError:
            prepareDBRes.error = textIntoFactsRes.error
            return prepareDBRes

        facts = textIntoFactsRes.facts

        # Проверка на количество фактов, надо ли загружать базу знаний вообще
        if len(facts) <= COUNT_NEAR_FIND_DB:
            self.isLoadDB = False
            self.factsForFindCollisions = facts
            return prepareDBRes

        # инициализация базы знаний
        if self.is_outer:
            self.RAGRetriever = RAGRetrieverGlobal(initDBGet(db_path=SAVE_DB_FILE_PATH))
        else:
            self.RAGRetriever = BaseRAGRetriever()

        # ФАКТЫ В ЭМБЕДДИНГИ
        factsIntoEmbeddingsRes = self.factsIntoEmbeddings(factsIntoEmbeddingsGet(facts))
        if factsIntoEmbeddingsRes.error.isError:
            prepareDBRes.error = factsIntoEmbeddingsRes.error
            return prepareDBRes

        # ЭМБЕДДИНГИ ЗАПИСЫВАЕМ В БАЗУ
        loadEmbeddingsDBRes = self.RAGRetriever.loadEmbeddingsDB(
            loadEmbeddingsDBGet(sentence_embeddings=factsIntoEmbeddingsRes.embeddings,
                                sentences=facts))
        if loadEmbeddingsDBRes.error.isError:
            prepareDBRes.error = loadEmbeddingsDBRes.error

        return prepareDBRes

    # ПРОВЕРКА КОЛЛИЗИИ ВОПРОСА С ФАКТАМИ ИЗ БАЗЫ ЗНАНИЙ
    def checkCollisionOne(self, getData: checkCollisionOneGet) -> checkCollisionOneResult:
        checkCollisionOneRes = checkCollisionOneResult(list(), ErrorClass(False, ""))
        factsIntoEmbeddingsRes = self.factsIntoEmbeddings(factsIntoEmbeddingsGet([getData.question]))
        if factsIntoEmbeddingsRes.error.isError:
            checkCollisionOneRes.error = factsIntoEmbeddingsRes.error
            return checkCollisionOneRes
        if self.isLoadDB:
            findTopNearestDBRes = self.RAGRetriever.findTopNearestDB(
                findTopNearestDBGet(query_embedding=factsIntoEmbeddingsRes.embeddings[0], k=COUNT_NEAR_FIND_DB, \
                                    NofNearestCellsToCheck=NOF_NEAREST_CELLS_TO_CHECK))
            if findTopNearestDBRes.error.isError:
                checkCollisionOneRes.error = findTopNearestDBRes.error
                return checkCollisionOneRes
            findTopNearestLLMRes = self.LLMService.findTopNearestLLM(
                findTopNearestLLMGet(question=getData.question, topFacts=findTopNearestDBRes.topNearest,
                                     countFind=COUNT_NUAR_FIND_LLM))
            if findTopNearestLLMRes.error.isError:
                checkCollisionOneRes.error = findTopNearestLLMRes.error
                return checkCollisionOneRes
            self.factsForFindCollisions = findTopNearestLLMRes.topNearest

        findCollisionsRes = self.LLMService.findCollisions(
            findCollisionsGet(question=getData.question, topFacts=self.factsForFindCollisions))
        if findCollisionsRes.error.isError:
            checkCollisionOneRes.error = findCollisionsRes.error
            return checkCollisionOneRes
        checkCollisionOneRes.arrCollisions = findCollisionsRes.arrCollisionResult
        return checkCollisionOneRes

    # ПРЕДОБРАБОТКА ТЕКСТА (ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ ДЛЯ СТАБИЛЬНОСТИ)
    def __textIntoFacts(self, getData: textIntoFactsGet) -> textIntoFactsResult:
        textIntoFactsRes = textIntoFactsResult(error=ErrorClass(False, ""), facts=[])
        # print("Полученный текст:", getData.text)

        # 1. разбивка на чанки
        splittingTextIntoChunksRes = self.PreprocessingDataService.splittingTextIntoChunks(
            splittingTextIntoChunksGet(text=getData.text, by_sentences=True, chunk_size=500, overlap=50))
        if splittingTextIntoChunksRes.error.isError:
            textIntoFactsRes.error = splittingTextIntoChunksRes.error
            return textIntoFactsRes
        chunks = splittingTextIntoChunksRes.chunks
        # print("Полученные чанки:", chunks)

        # 2. очистка текста (чанков по отдельности) - ПОСЛЕДОВАТЕЛЬНО
        cleaned_chunks = []
        for chunk in chunks:
            cleanTextRes = self.PreprocessingDataService.cleanText(cleanTextGet(text=chunk))
            cleaned_chunks.append(cleanTextRes.text)
        # print("Очищенные чанки:", cleaned_chunks)

        # 3. выделение из чанков фактов - ПОСЛЕДОВАТЕЛЬНО
        all_facts = []
        for chunk in cleaned_chunks:
            splittingChunksIntoFactsRes = self.LLMService.splittingChunksIntoFacts(
                splittingChunksIntoFactsGet(chunk=chunk))
            if splittingChunksIntoFactsRes.error.isError:
                textIntoFactsRes.error = splittingChunksIntoFactsRes.error
                return textIntoFactsRes
            all_facts.extend(splittingChunksIntoFactsRes.arrFacts)

        textIntoFactsRes.facts = all_facts
        # print("Полученные факты:", textIntoFactsRes.facts)

        return textIntoFactsRes

    # Преобразует список предложений в массив эмбеддингов
    def factsIntoEmbeddings(self, getData: factsIntoEmbeddingsGet) -> factsIntoEmbeddingsResult:
        factsIntoEmbeddingsRes = factsIntoEmbeddingsResult(None, ErrorClass(False, ""))
        try:
            # можно добавить show_progress_bar=True для наглядности
            factsIntoEmbeddingsRes.embeddings = self.embedding_model.encode(getData.sentences)
        except Exception as e:
            factsIntoEmbeddingsRes.error = ErrorClass(True, f"Ошибка генерации эмбеддингов: {str(e)}")
        return factsIntoEmbeddingsRes


# Этот блок __main__ здесь не используется при запуске из testOnWikiDataset.py,
# но я оставлю его на случай, если вы захотите запустить этот файл напрямую для отладки.
if __name__ == "__main__":
    # Для прямого запуска этого файла потребуется создать модель здесь
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    service = AnticollisionMainClass(embedding_model=embedding_model)

    try:
        text = open("../file/test_collision_text.txt", "r", encoding="utf-8").read()
        question = open("../file/test_collision_question.txt", "r", encoding="utf-8").read()
    except FileNotFoundError as e:
        print(f"Ошибка чтения файла: {e}")
        exit(1)

    prepareDBRes = service.prepareDB(prepareDBGet(text=text))
    if prepareDBRes.error.isError:
        print(prepareDBRes.error.messageError)
        exit(1)

    checkCollisionOneRes = service.checkCollisionOne(checkCollisionOneGet(question=question))
    if checkCollisionOneRes.error.isError:
        print(checkCollisionOneRes.error.messageError)
        exit(2)

    print("Ответ модели (коллизии):")
    for i, ans in enumerate(checkCollisionOneRes.arrCollisions, 1):
        print(f"{i}: {ans}")