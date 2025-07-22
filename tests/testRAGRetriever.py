# test_rag_retriever.py
import unittest
import numpy as np
import os
import shutil
import json

# Импортируем классы из ваших файлов
from rag.database.RAGRetriever import BaseRAGRetriever, RAGRetrieverGlobal
from rag.entity.dataDB import (initDBGet, loadEmbeddingsDBGet, findTopNearestDBGet,
                               loadDBGet, saveDBGet)


class TestBaseRAGRetriever(unittest.TestCase):

    def setUp(self):
        """Вызывается перед каждым тестовым методом."""
        self.retriever = BaseRAGRetriever()
        self.sentences = ["кошка сидит на коврике", "собака играет с мячом", "попугай ест зерно"]
        # Создаем простые, различимые эмбеддинги
        self.embeddings = np.array([
            [1.0, 0.0, 0.0],  # Вектор для "кошки"
            [0.0, 1.0, 0.0],  # Вектор для "собаки"
            [0.0, 0.0, 1.0]   # Вектор для "попугая"
        ]).astype('float32')

    def test_build_index_and_search(self):
        """
        Тест: успешное построение индекса и поиск ближайшего предложения.
        """
        # 1. Построение индекса
        getData = loadEmbeddingsDBGet(sentence_embeddings=self.embeddings, sentences=self.sentences)
        result = self.retriever.loadEmbeddingsDB(getData)
        self.assertFalse(result.error.isError)
        self.assertTrue(self.retriever._is_ready)
        self.assertEqual(self.retriever.index.ntotal, 3)

        # 2. Поиск
        # Создаем вектор запроса, максимально близкий к "кошке"
        query_embedding = np.array([0.9, 0.1, 0.0]).astype('float32')
        search_data = findTopNearestDBGet(query_embedding=query_embedding, k=1)
        search_result = self.retriever.findTopNearestDB(search_data)

        # 3. Проверка результата
        self.assertFalse(search_result.error.isError)
        self.assertEqual(len(search_result.topNearest), 1)
        self.assertEqual(search_result.topNearest[0], "кошка сидит на коврике")

    def test_search_before_build(self):
        """
        Тест: попытка поиска до построения индекса должна возвращать ошибку.
        """
        query_embedding = np.array([1.0, 0.0, 0.0]).astype('float32')
        search_data = findTopNearestDBGet(query_embedding=query_embedding, k=1)
        result = self.retriever.findTopNearestDB(search_data)
        self.assertTrue(result.error.isError)
        self.assertIn("Индекс не построен", result.error.messageError)

    def test_build_with_mismatched_data(self):
        """
        Тест: построение индекса с разным количеством эмбеддингов и предложений.
        """
        mismatched_embeddings = np.array([[1.0, 0.0, 0.0]]).astype('float32') # 1 эмбеддинг
        getData = loadEmbeddingsDBGet(sentence_embeddings=mismatched_embeddings, sentences=self.sentences) # 3 предложения
        result = self.retriever.loadEmbeddingsDB(getData)
        self.assertTrue(result.error.isError)
        self.assertIn("Количество эмбеддингов должно совпадать", result.error.messageError)


class TestRAGRetrieverGlobal(unittest.TestCase):

    def setUp(self):
        """Создает временную директорию для БД перед каждым тестом."""
        self.test_db_path = r".\test_db_dir"
        os.makedirs(self.test_db_path, exist_ok=True)
        # Подготовим два набора данных: маленький и большой
        self.small_sentences = ["A", "B", "C"]
        self.small_embeddings = np.random.rand(3, 128).astype('float32')

        # Большой набор данных для проверки IndexIVFFlat
        self.large_sentences = [f"Sentence {i}" for i in range(1050)]
        self.large_embeddings = np.random.rand(1050, 128).astype('float32')


    def tearDown(self):
        """Удаляет временную директорию после каждого теста."""
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    def test_init_and_create_db(self):
        """
        Тест: инициализация, построение индекса и его автоматическое сохранение на диск.
        """
        # 1. Инициализация
        retriever = RAGRetrieverGlobal(initDBGet(db_path=self.test_db_path))
        self.assertFalse(retriever._is_db_exist) # Базы еще нет

        # 2. Построение индекса (с маленьким набором данных, создастся IndexFlatIP)
        getData = loadEmbeddingsDBGet(sentence_embeddings=self.small_embeddings, sentences=self.small_sentences)
        result = retriever.loadEmbeddingsDB(getData)
        self.assertFalse(result.error.isError)
        self.assertTrue(retriever._is_ready)

        # 3. Проверка, что файлы были сохранены
        self.assertTrue(os.path.exists(os.path.join(self.test_db_path, "knowledge_base.faiss")))
        self.assertTrue(os.path.exists(os.path.join(self.test_db_path, "sentences.json")))
        self.assertTrue(retriever._is_db_exist)

    def test_save_and_load_db(self):
        """
        Тест: ручное сохранение и последующая загрузка индекса.
        """
        # --- Фаза 1: Создание и сохранение ---
        retriever1 = RAGRetrieverGlobal(initDBGet(db_path=self.test_db_path))
        getData = loadEmbeddingsDBGet(sentence_embeddings=self.small_embeddings, sentences=self.small_sentences)
        retriever1.loadEmbeddingsDB(getData)
        # Допустим, мы явно вызываем saveDB
        save_result = retriever1.saveDB(saveDBGet())
        self.assertFalse(save_result.error.isError)

        # --- Фаза 2: Загрузка в новый экземпляр ---
        retriever2 = RAGRetrieverGlobal(initDBGet(db_path=self.test_db_path))
        # При инициализации он должен был автоматически загрузить базу
        self.assertTrue(retriever2._is_ready)
        self.assertEqual(retriever2.index.ntotal, len(self.small_sentences))
        self.assertEqual(retriever2.original_sentences, self.small_sentences)

    def test_approximate_index_creation(self):
        """
        Тест: проверяет, что для большого объема данных создается приблизительный индекс (IndexIVFFlat).
        """
        retriever = RAGRetrieverGlobal(initDBGet(db_path=self.test_db_path))
        getData = loadEmbeddingsDBGet(sentence_embeddings=self.large_embeddings, sentences=self.large_sentences)
        retriever.loadEmbeddingsDB(getData)

        # Проверяем тип созданного индекса
        # IndexIVFFlat в Python - это faiss.swigfaiss.IndexIVFFlat
        self.assertIn("IndexIVFFlat", str(type(retriever.index)))
        self.assertEqual(retriever.index.ntotal, len(self.large_sentences))

    def test_load_non_existent_db(self):
        """
        Тест: попытка загрузить несуществующую БД не должна вызывать ошибку при инициализации,
        но должна корректно устанавливать флаги.
        """
        # Указываем путь, где точно ничего нет
        non_existent_path = "./non_existent_dir_12345"
        retriever = RAGRetrieverGlobal(initDBGet(db_path=non_existent_path))
        self.assertFalse(retriever._is_ready)
        self.assertFalse(retriever._is_db_exist)
        # Убедимся, что директория не была создана при инициализации
        self.assertFalse(os.path.exists(non_existent_path))

    def test_overridden_search_method(self):
        """
        Тест: проверяет, что переопределенный метод findTopNearestDB работает.
        """
        retriever = RAGRetrieverGlobal(initDBGet(db_path=self.test_db_path))
        getData = loadEmbeddingsDBGet(sentence_embeddings=self.large_embeddings, sentences=self.large_sentences)
        retriever.loadEmbeddingsDB(getData)

        query_embedding = np.random.rand(128).astype('float32')
        # Ищем с параметром nprobe (NofNearestCellsToCheck)
        search_data = findTopNearestDBGet(query_embedding=query_embedding, k=5, NofNearestCellsToCheck=8)
        result = retriever.findTopNearestDB(search_data)

        self.assertFalse(result.error.isError)
        self.assertEqual(len(result.topNearest), 5)


if __name__ == '__main__':
    unittest.main()
