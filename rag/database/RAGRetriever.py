import numpy as np
import faiss
from typing import List, Dict, Union
import os
import json
from rag.entity.dataDB import (ErrorClass, findTopNearestDBResult, findTopNearestDBGet,
                               loadEmbeddingsDBResult, loadEmbeddingsDBGet, saveDBResult,
                               saveDBGet, loadDBGet, loadDBResult, initDBGet)


class BaseRAGRetriever:
    """
    Базовый класс для RAG-ретривера, работающего в памяти.
    Использует косинусное сходство через полный перебор (brute-force).
    """
    def __init__(self):
        self.index = None
        self.original_sentences: List[str] = []
        self._is_ready: bool = False

    def loadEmbeddingsDB(self, getData: loadEmbeddingsDBGet) -> loadEmbeddingsDBResult:
        error = ErrorClass(isError=False, messageError="")
        if getData.sentence_embeddings.shape[0] != len(getData.sentences):
            error.isError = True
            error.messageError = "Количество эмбеддингов должно совпадать с количеством предложений."

        embedding_dim = getData.sentence_embeddings.shape[1]
        # Используем IndexFlatIP для точного поиска по скалярному произведению
        self.index = faiss.IndexFlatIP(embedding_dim)

        sentence_embeddings_normalized = getData.sentence_embeddings.astype('float32')
        faiss.normalize_L2(sentence_embeddings_normalized)

        self.index.add(sentence_embeddings_normalized)
        self.original_sentences = getData.sentences
        self._is_ready = True
        print(f"Индекс (brute-force) успешно построен. Добавлено {self.index.ntotal} векторов.")
        return loadEmbeddingsDBResult(error=error)

    def findTopNearestDB(self, getData: findTopNearestDBGet) -> findTopNearestDBResult:
        """
       Выполняет поиск k-наиболее релевантных предложений для заданного эмбеддинга факта.
       Args:
           getData.query_embedding (np.ndarray): 1D-массив NumPy с эмбеддингом запроса (факта).
           getData.k (int): Количество наиболее релевантнх предложений для возврата (top-k).
       Returns:
           List[Dict[str, Union[str, float]]]:
               Список словарей, где каждый словарь содержит:
               - 'sentence': найденное релевантное предложение (str)
               - 'similarity': косинусная близость (float)
       Raises:
           Exception: Если индекс еще не был построен с помощью метода build_index.
       """
        error = ErrorClass(isError=False, messageError="")

        if not self._is_ready or self.index is None:
            error.isError = True
            error.messageError = "Индекс не построен. Пожалуйста, вызовите \
                метод .build_index() перед поиском."

        k = getData.k
        if getData.k > self.index.ntotal:
            k = self.index.ntotal

        query_vector = np.array([getData.query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)

        # scores - это значения косинусного сходства, а не расстояния
        scores, indices = self.index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append(self.original_sentences[idx])
        return findTopNearestDBResult(topNearest=results, error=error)


class RAGRetrieverGlobal(BaseRAGRetriever):
    """
    Улучшенный RAG-ретривер, который:
    1. Наследуется от BaseRAGRetriever.
    2. Использует **приблизительный поиск (ANN)** для высокой скорости.
    3. Поддерживает сохранение и загрузку индекса с диска.
    """

    def __init__(self, getData: initDBGet):
        """
        Инициализирует ретривер, указывая путь для хранения базы знаний.
        Args:
            getData.db_path (str): Путь к директории для сохранения/загрузки индекса.
                На мой взгляд сейчас должен иметь вид: .\DataBases\directory.
                Если там нет файлов (knowledge_base.faiss, sentences.json),
                то после вызова метода 'self.build_index' они появятся.
        """

        super().__init__()
        self.db_path: str = getData.db_path
        self._is_db_exist: bool = False
        self.index_file = os.path.join(self.db_path, "knowledge_base.faiss")
        self.sentences_file = os.path.join(self.db_path, "sentences.json")

        # Попробуем загрузить индекс, если он уже существует
        if os.path.exists(self.db_path):
            try:
                self.loadDB(loadDBGet())
                self._is_db_exist = True
            except FileNotFoundError:
                self._is_db_exist = False

    def loadEmbeddingsDB(self, getData: loadEmbeddingsDBGet) -> loadEmbeddingsDBResult:
        """
        **Переопределенный метод.**
        Строит **приблизительный** FAISS-индекс (IndexIVFFlat) для оптимизации скорости
        и сохраняет его.
        """
        error = ErrorClass(isError=False, messageError="")
        print("--- Построение приблизительного индекса (IndexIVFFlat) ---")
        if getData.sentence_embeddings.shape[0] != len(getData.sentences):
            error.isError = True
            error.messageError = "Количество эмбеддингов должно совпадать с количеством предложений."
            return loadEmbeddingsDBResult(error=error)

        num_embeddings = getData.sentence_embeddings.shape[0]
        embedding_dim = getData.sentence_embeddings.shape[1]

        sentence_embeddings_normalized = getData.sentence_embeddings.astype('float32')
        faiss.normalize_L2(sentence_embeddings_normalized)

        MIN_VECTORS_FOR_IVF = 1000

        if num_embeddings < MIN_VECTORS_FOR_IVF:
            print(f"--- Данных мало ({num_embeddings} векторов). Используется точный индекс (IndexFlatIP) ---")
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.index.add(sentence_embeddings_normalized)
        else:
            print(f"--- Данных много ({num_embeddings} векторов). Строится приблизительный индекс (IndexIVFFlat) ---")
            nlist = min(100, int(4 * np.sqrt(num_embeddings)))
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)

            print(f"Тренировка индекса на {num_embeddings} векторах...")
            self.index.train(sentence_embeddings_normalized)
            self.index.add(sentence_embeddings_normalized)

        self.original_sentences = getData.sentences
        self._is_ready = True

        self.saveDB(saveDBGet())

        return loadEmbeddingsDBResult(error=error)

    def findTopNearestDB(self, getData: findTopNearestDBGet) -> findTopNearestDBResult:
        """
        **Переопределенный метод.**
        Выполняет поиск с параметром NofNearestCellsToCheck для управления компромиссом скорость/точность.
        Args:
            getData.query_embedding (np.ndarray): Эмбеддинг запроса.
            getData.k (int): Количество результатов для возврата.
            getData.NofNearestCellsToCheck (int): Количество ячеек (кластеров) для поиска. Чем выше, тем точнее и медленнее.
        """
        if not self._is_ready:
            error = ErrorClass(isError=True, messageError="Индекс не загружен и не построен.")
            return findTopNearestDBResult(topNearest=[], error=error)

        # Устанавливаем, сколько ближайших ячеек проверять при поиске
        self.index.NofNearestCellsToCheck = getData.NofNearestCellsToCheck
        # Вызываем оригинальный метод search из родительского класса
        return super().findTopNearestDB(getData)

    def saveDB(self, _: saveDBGet) -> saveDBResult:
        """Сохраняет индекс и предложения на диск."""
        error = ErrorClass(isError=False, messageError="")
        if not self._is_ready:
            error.isError = True
            error.messageError = "Нечего сохранять. Индекс не был построен."

        print(f"Сохранение индекса и предложений в '{self.db_path}'...")
        os.makedirs(self.db_path, exist_ok=True)
        index_file = os.path.join(self.db_path, "knowledge_base.faiss")
        sentences_file = os.path.join(self.db_path, "sentences.json")

        faiss.write_index(self.index, index_file)
        with open(sentences_file, 'w', encoding='utf-8') as f:
            json.dump(self.original_sentences, f, ensure_ascii=False, indent=4)

        self._is_db_exist = True
        return saveDBResult(error=error)

    def loadDB(self, getData: loadDBGet) -> loadDBResult:
        """Загружает индекс и предложения с диска."""
        index_file = self.index_file
        sentences_file = self.sentences_file
        if getData.db_path is not None:
            index_file = os.path.join(getData.db_path, "knowledge_base.faiss")
            sentences_file = os.path.join(getData.db_path, "sentences.json")

        error = ErrorClass(isError=False, messageError="")
        if not os.path.exists(index_file) or not os.path.exists(sentences_file):
            error.isError = True
            error.messageError = "Файлы индекса или предложений не найдены."

        print(f"Загрузка индекса из '{self.db_path}'...")
        self.index = faiss.read_index(index_file)
        with open(sentences_file, 'r', encoding='utf-8') as f:
            self.original_sentences = json.load(f)

        self._is_ready = True
        print(f"Ретривер успешно загружен. В индексе {self.index.ntotal} векторов.")
        return loadDBResult(error=error)
