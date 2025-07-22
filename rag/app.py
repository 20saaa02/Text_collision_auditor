# Основной класс запуска RAG
from rag.llm.LLMService import LLMService

class AnticollisionMainClass:
    def __init__(self):
        self.LLMService = LLMService()

    def process(self, text):
        pass
        # TODO



if __name__ == "__main__":
    # инициализация
    service = AnticollisionMainClass()

    # получить текст для проверки

    # обработать текст, получить ответ
    ans = service.process(text)

    # обработать ответ