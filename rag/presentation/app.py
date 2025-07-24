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
    factsIntoEmbeddingsGet, factsIntoEmbeddingsResult, checkCollisionOneGet, checkCollisionOneResult, checkCollisionAllGet, checkCollisionAllResult
from entity.dataDB import findTopNearestDBGet, loadEmbeddingsDBGet, initDBGet

SAVE_DB_FILE_PATH = "../file/database/save_db.txt"
COUNT_NEAR_FIND_DB = 30
COUNT_NUAR_FIND_LLM = 5
NOF_NEAREST_CELLS_TO_CHECK = 10

# Для прямого запуска этого файла потребуется создать модель здесь
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# Основной класс запуска RAG
class AnticollisionMainClass:
    def __init__(self, is_outer: bool = False):
        self.RAGRetriever = None
        self.LLMService = LLMService(initLLMServiceGet())
        self.PreprocessingDataService = PreprocessingDataService(initPreprocessingDataServiceGet())
        # Инициализация модели для эмбеддингов
        #self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Легкая модель
        # Или для русского языка: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
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
        #print("Результат факты:", textIntoFactsRes.facts[5:])

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
            # ПОИСК 30 БЛИЖАЙШИХ ПО БАЗЕ ЗНАНИЙ
            findTopNearestDBRes = self.RAGRetriever.findTopNearestDB(
                findTopNearestDBGet(query_embedding=factsIntoEmbeddingsRes.embeddings[0], k=COUNT_NEAR_FIND_DB, \
                                    NofNearestCellsToCheck=NOF_NEAREST_CELLS_TO_CHECK))
            if findTopNearestDBRes.error.isError:
                checkCollisionOneRes.error = findTopNearestDBRes.error
                return checkCollisionOneRes
            #print("Топ 30 фактов:", findTopNearestDBRes.topNearest)
            # ПОИСК 5 БЛИЖАЙШИХ ИЗ 30 ЧЕРЕЗ ЛЛМ
            findTopNearestLLMRes = self.LLMService.findTopNearestLLM(
                findTopNearestLLMGet(question=getData.question, topFacts=findTopNearestDBRes.topNearest,
                                     countFind=COUNT_NUAR_FIND_LLM))
            if findTopNearestLLMRes.error.isError:
                checkCollisionOneRes.error = findTopNearestLLMRes.error
                return checkCollisionOneRes
            self.factsForFindCollisions = findTopNearestLLMRes.topNearest

        #print("Топ 5 фактов:", findTopNearestLLMRes.topNearest)
        # ЛЛМ ИЩЕТ КОЛЛИЗИИ
        findCollisionsRes = self.LLMService.findCollisions(
            findCollisionsGet(question=getData.question, topFacts=self.factsForFindCollisions))
        
        if findCollisionsRes.error.isError:
            checkCollisionOneRes.error = findCollisionsRes.error
            return checkCollisionOneRes
        checkCollisionOneRes.arrCollisions = findCollisionsRes.arrCollisionResult
        return checkCollisionOneRes

    # ПРОВЕРКА КОЛЛИЗИИ ПО ВСЕМ ФАКТАМ БАЗЫ ЗНАНИЙ
    def checkCollisionAll(self, _: checkCollisionAllGet) -> checkCollisionAllResult:
        checkCollisionAllRes = checkCollisionAllResult(list(), ErrorClass(False, ""))
        if self.isLoadDB:
            returnAllFactsFromDBRes = self.RAGRetriever.returnAllFactsFromDB(returnAllFactsFromDBGet())
            if returnAllFactsFromDBRes.error.isError:
                checkCollisionAllRes.error = returnAllFactsFromDBRes.error
                return checkCollisionAllRes
            facts = returnAllFactsFromDBRes.original_facts
        else:
            facts = self.factsForFindCollisions
        
        #print("Факты:", facts)
        for query_fact in facts:
            checkCollisionOneRes = self.checkCollisionOne(checkCollisionOneGet(question=query_fact))
            if checkCollisionOneRes.error.isError:
                checkCollisionAllRes.error = checkCollisionOneRes.error
                break
            if len(checkCollisionOneRes.arrCollisions) > 0:
                checkCollisionOneQueryRes = checkCollisionOneQueryResult(question=query_fact, arrCollisions=checkCollisionOneRes.arrCollisions)
                checkCollisionAllRes.arrCollisionsQueryRes.append(checkCollisionOneQueryRes)
        return checkCollisionAllRes


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

        # очистка списка предложений
        sentences = list()
        for i in range(len(getData.sentences)):
            cleanTextRes = self.PreprocessingDataService.cleanText(cleanTextGet(text = getData.sentences[i]))
            sentences.append(cleanTextRes.text)
        
        # преобразование очищенного списка предложений в эмбеддинги
        try:
            # можно добавить show_progress_bar=True для наглядности
            factsIntoEmbeddingsRes.embeddings = self.embedding_model.encode(sentences)
        except Exception as e:
            factsIntoEmbeddingsRes.error = ErrorClass(True, f"Ошибка генерации эмбеддингов: {str(e)}")
        return factsIntoEmbeddingsRes


# Этот блок __main__ здесь не используется при запуске из testOnWikiDataset.py,
# но я оставлю его на случай, если вы захотите запустить этот файл напрямую для отладки.
if __name__ == "__main__":

    service = AnticollisionMainClass()

    try:
        text = open("../file/test_collision_text.txt", "r", encoding="utf-8").read()
        question = open("../file/test_collision_question.txt", "r", encoding="utf-8").read()

        text_collis_in = open("../file/test_collision_in_text.txt", "r", encoding="utf-8").read()
    except FileNotFoundError as e:
        print(f"Ошибка чтения файла: {e}")
        exit(1)

    prepareDBRes = service.prepareDB(prepareDBGet(text=text_collis_in, loadDBbtw=False))
    if prepareDBRes.error.isError:
        print(prepareDBRes.error.messageError)
        exit(1)

    # внешний запрос
    #checkCollisionOneRes = service.checkCollisionOne(checkCollisionOneGet(question=question))
    #if checkCollisionOneRes.error.isError:
    #    print(checkCollisionOneRes.error.messageError)
    #    exit(2)
    #print("Ответ модели (коллизии):")
    #for i, ans in enumerate(checkCollisionOneRes.arrCollisions, 1):
    #    print(f"{i}: {ans}")

    # внутренный запрос
    checkCollisionAllRes = service.checkCollisionAll(checkCollisionAllGet())
    if checkCollisionAllRes.error.isError:
        print(checkCollisionAllRes.error.messageError)
        exit(3)
    for res in checkCollisionAllRes.arrCollisionsQueryRes:
        print("Найдено противоречие:", res.question, res.arrCollisions)