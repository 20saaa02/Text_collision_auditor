import numpy as np
import faiss
from typing import List
import os
import json
from rag.entity.dataDB import (ErrorClass, findTopNearestDBResult, findTopNearestDBGet,
                               loadEmbeddingsDBResult, loadEmbeddingsDBGet, saveDBResult,
                               saveDBGet, loadDBGet, loadDBResult, initDBGet)


class BaseRAGRetriever:
    """
    Базовый класс для RAG-ретривера, работающего в памяти.

    Реализует основную логику создания FAISS-индекса и поиска в нем
    без сохранения на диск. Использует точный поиск (brute-force) по метрике L2.
    """

    def __init__(self):
        """Инициализирует атрибуты базового ретривера."""
        self.index = None
        self.original_sentences: List[str] = []
        self._is_ready: bool = False

    def loadEmbeddingsDB(self, getData: loadEmbeddingsDBGet) -> loadEmbeddingsDBResult:
        """
        Загружает эмбеддинги и предложения, строит FAISS-индекс в памяти.
        Args:
            getData.sentence_embeddings (np.ndarray): Массив NumPy с эмбеддингами предложений.
            getData.sentences (List[str]): Список оригинальных текстовых предложений.
        Returns:
            loadEmbeddingsDBResult:
                Объект результата, содержащий:
                - error (ErrorClass): Объект ошибки. `isError` будет True, если количество
                  эмбеддингов не совпадает с количеством предложений.
        """
        error = ErrorClass(isError=False, messageError="")
        if getData.sentence_embeddings.shape[0] != len(getData.sentences):
            error.isError = True
            error.messageError = "Количество эмбеддингов должно совпадать с количеством предложений."
            return loadEmbeddingsDBResult(error=error)

        embedding_dim = getData.sentence_embeddings.shape[1]

        self.index = faiss.IndexFlatL2(embedding_dim)

        sentence_embeddings_normalized = getData.sentence_embeddings.astype('float32')
        faiss.normalize_L2(sentence_embeddings_normalized)

        self.index.add(sentence_embeddings_normalized)
        self.original_sentences = getData.sentences
        self._is_ready = True
        print(f"Индекс (brute-force, L2) успешно построен. Добавлено {self.index.ntotal} векторов.")
        return loadEmbeddingsDBResult(error=error)

    def findTopNearestDB(self, getData: findTopNearestDBGet) -> findTopNearestDBResult:
        """
        Находит k наиболее близких предложений к заданному вектору запроса.
        Args:
            getData.query_embedding (np.ndarray): Векторное представление (эмбеддинг) поискового запроса.
            getData.k (int): Желаемое количество ближайших соседей для поиска (top-k).
            getData.NofNearestCellsToCheck (int): Параметр для приблизительных индексов (здесь не используется).
        Returns:
            findTopNearestDBResult:
                Объект результата, содержащий:
                - topNearest (List[str]): Список найденных наиболее релевантных предложений.
                - error (ErrorClass): Объект ошибки. `isError` будет True, если индекс
                  не был предварительно построен.
        """
        error = ErrorClass(isError=False, messageError="")

        if not self._is_ready or self.index is None:
            error.isError = True
            error.messageError = "Индекс не построен. Пожалуйста, вызовите " \
                                 "метод .loadEmbeddingsDB() перед поиском."
            return findTopNearestDBResult(topNearest=[], error=error)

        k = getData.k
        if getData.k > self.index.ntotal:
            k = self.index.ntotal

        query_vector = np.array([getData.query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)

        # Метод search для L2 возвращает расстояния, а не сходства. Чем меньше, тем лучше.
        distances, indices = self.index.search(query_vector, k)

        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.original_sentences[idx])
        return findTopNearestDBResult(topNearest=results, error=error)


class RAGRetrieverGlobal(BaseRAGRetriever):
    """
    Расширенный RAG-ретривер с возможностью сохранения и загрузки индекса.

    Автоматически выбирает тип индекса (точный или приблизительный) в зависимости
    от объема данных и управляет персистентностью базы знаний.
    """

    def __init__(self, getData: initDBGet):
        """
        Инициализирует ретривер, задает пути и пытается загрузить существующую БД.
        Args:
            getData.db_path (str): Путь к директории для хранения или загрузки
                                   базы данных (индекса и предложений).
        """
        super().__init__()
        self.db_path: str = getData.db_path
        self._is_db_exist: bool = False
        self._is_ready: bool = False
        self.index_file = os.path.join(self.db_path, "knowledge_base.faiss")
        self.sentences_file = os.path.join(self.db_path, "sentences.json")

        if os.path.exists(self.index_file) and os.path.exists(self.sentences_file):
            load_result = self.loadDB(loadDBGet())
            if not load_result.error.isError:
                self._is_db_exist = True

    def loadEmbeddingsDB(self, getData: loadEmbeddingsDBGet) -> loadEmbeddingsDBResult:
        """
        Создает или перезаписывает базу данных, строя FAISS-индекс и сохраняя его.
        Args:
            getData.sentence_embeddings (np.ndarray): Массив NumPy с эмбеддингами.
            getData.sentences (List[str]): Список оригинальных текстовых предложений.
        Returns:
            loadEmbeddingsDBResult:
                Объект результата, содержащий:
                - error (ErrorClass): Объект ошибки. `isError` будет True, если
                  размеры данных не совпадают.
        """
        error = ErrorClass(isError=False, messageError="")
        print("--- Построение FAISS-индекса с метрикой L2 ---")
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
            print(f"--- Данных мало ({num_embeddings} векторов). Используется точный индекс (IndexFlatL2) ---")
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            print(
                f"--- Данных много ({num_embeddings} векторов). Строится приблизительный индекс (IndexIVFFlat, L2) ---")
            nlist = min(100, int(4 * np.sqrt(num_embeddings)))
            # --- ИЗМЕНЕНИЕ №3: Квантизатор тоже должен использовать L2 ---
            quantizer = faiss.IndexFlatL2(embedding_dim)
            # --- ИЗМЕНЕНИЕ №4: Основной индекс также использует L2 ---
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)

            print(f"Тренировка индекса на {num_embeddings} векторах...")
            self.index.train(sentence_embeddings_normalized)

        self.index.add(sentence_embeddings_normalized)
        self.original_sentences = getData.sentences
        self._is_ready = True
        self.saveDB(saveDBGet())

        return loadEmbeddingsDBResult(error=error)

    def findTopNearestDB(self, getData: findTopNearestDBGet) -> findTopNearestDBResult:
        """
        Выполняет поиск k ближайших соседей с учетом типа индекса.
        Args:
            getData.query_embedding (np.ndarray): Эмбеддинг поискового запроса.
            getData.k (int): Желаемое количество ближайших соседей.
            getData.NofNearestCellsToCheck (int): Количество ближайших кластеров (ячеек)
                для проверки в приблизительном индексе (IVF). Увеличивает точность
                за счет производительности.
        Returns:
            findTopNearestDBResult:
                Объект результата, содержащий:
                - topNearest (List[str]): Список найденных предложений.
                - error (ErrorClass): Объект ошибки.
        """
        if not self._is_ready or not hasattr(self.index, 'nprobe'):
            # Если это простой IndexFlatIP, у него нет nprobe, просто вызываем родительский метод
            return super().findTopNearestDB(getData)

        # Устанавливаем, сколько ближайших ячеек проверять при поиске
        self.index.nprobe = getData.NofNearestCellsToCheck
        return super().findTopNearestDB(getData)

    def saveDB(self, _: saveDBGet) -> saveDBResult:
        """
        Сохраняет текущий FAISS-индекс и список предложений в файлы на диске.
        Args:
            _ (saveDBGet): Параметры не требуются, метод использует внутреннее состояние объекта.
        Returns:
            saveDBResult:
                Объект результата, содержащий:
                - error (ErrorClass): Объект ошибки. `isError` будет True, если
                  индекс не был построен и сохранять нечего.
        """
        error = ErrorClass(isError=False, messageError="")
        if not self._is_ready or self.index is None:
            error.isError = True
            error.messageError = "Нечего сохранять. Индекс не был построен."
            return saveDBResult(error=error)

        print(f"Сохранение индекса и предложений в '{self.db_path}'...")
        os.makedirs(self.db_path, exist_ok=True)

        faiss.write_index(self.index, self.index_file)
        with open(self.sentences_file, 'w', encoding='utf-8') as f:
            json.dump(self.original_sentences, f, ensure_ascii=False, indent=4)

        self._is_db_exist = True
        return saveDBResult(error=error)

    def loadDB(self, getData: loadDBGet) -> loadDBResult:
        """
        Загружает FAISS-индекс и список предложений из файлов на диске.
        Args:
            getData.db_path (str, optional): Путь к директории. В текущей реализации
                этот аргумент не используется, т.к. путь берется из `self.db_path`,
                установленного при инициализации.
        Returns:
            loadDBResult:
                Объект результата, содержащий:
                - error (ErrorClass): Объект ошибки. `isError` будет True, если файлы
                  не найдены или повреждены.
        """
        error = ErrorClass(isError=False, messageError="")

        if not os.path.exists(self.index_file) or not os.path.exists(self.sentences_file):
            error.isError = True
            error.messageError = f"Файлы для загрузки не найдены в '{self.db_path}'."
            return loadDBResult(error=error)

        print(f"Загрузка индекса из '{self.db_path}'...")
        # Проверяем загрузку
        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.sentences_file, 'r', encoding='utf-8') as f:
                self.original_sentences = json.load(f)

            self._is_ready = True
            print(f"Ретривер успешно загружен. В индексе {self.index.ntotal} векторов.")

        except (RuntimeError, Exception) as e:
            # Ловим ошибку faiss (RuntimeError) и любые другие
            error.isError = True
            error.messageError = f"Не удалось прочитать файлы базы данных: {e}"
            self._is_ready = False
            self.index = None

        return loadDBResult(error=error)
