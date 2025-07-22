from service.MyLLM import MyLLM
from entity.dataLLM import ErrorClass, findTopNearestLLMGet, findTopNearestLLMResult, findCollisionsResult, findCollisionsGet, \
    splittingChunksIntoFactsGet, splittingChunksIntoFactsResult, initLLMServiceGet
    

class LLMService:
    def __init__(self, _: initLLMServiceGet):
        self.llm = MyLLM()

    # выделить из топ-N топ-K ближайших по смыслу к question
    def findTopNearestLLM(self, getData: findTopNearestLLMGet) -> findTopNearestLLMResult:

        error = ErrorClass(False, "findTopNearestLLM ")
        topNearest = list()

        prompt = (
            f"Ты должен выбрать {getData.countFind} факта из списка, которые наиболее точно соответствуют смыслу вопроса.\n"
            f"Вопрос: {getData.question}\n"
            f"Список фактов:\n" + "\n".join(f"- {fact}" for fact in getData.topFacts) + "\n\n"
            "Требования к ответу:\n"
            "1. Выбери только факты из приведенного списка\n"
            "2. Не изменяй формулировки фактов\n"
            "3. Не добавляй свои комментарии\n"
            "4. Перечисли факты в порядке убывания релевантности\n"
            "5. Разделяй факты строго тремя дефисами (---)\n\n"
            "Формат ответа:\n"
            "<точная цитата факта 1> --- <точная цитата факта 2> --- ... --- <точная цитата факта N>"
        )

        specialization = "Ты - эксперт по сравнению фактов по смыслу."

        try:
            response = self.llm.ask(prompt, specialization)
            
            # Парсинг ответа с разделителем "---"
            try:
                # Разбиваем строку по разделителю "---"
                topNearest = [fact.strip() for fact in response.split('---')]
                # Обрезаем до нужного количества фактов, если вдруг пришло больше
                topNearest = topNearest[:getData.countFind]
                
                # Проверяем, что получили нужное количество фактов
                if len(topNearest) < getData.countFind:
                    error.isError = True
                    error.messageError += f"Получено только {len(topNearest)} фактов из запрошенных {getData.countFind}"
                
            except Exception as e:
                error.isError = True
                error.messageError += f"Ошибка парсинга ответа: {str(e)}. Ответ LLM: {response}"
                
        except Exception as e:
            error.isError = True
            error.messageError += f"Ошибка при запросе к LLM: {str(e)}"

        return findTopNearestLLMResult(topNearest, error)
    

    def findCollisions(self, getData: findCollisionsGet) -> findCollisionsResult:
        error = ErrorClass(False, "findCollisions ")
        arrCollisionResult = []

        specialization = "Ты - эксперт по анализу фактов на предмет соответствия вопросу."

        prompt = f"""
            Проанализируй каждый факт из списка относительно вопроса и верни ТОЛЬКО те, которые противоречат вопросу (не могут быть истинными одновременно с вопросом).

            Если таких фактов нет - верни ПУСТУЮ СТРОКУ.

            Формат ответа: 
            - Если есть коллизии: "факт1 --- факт2 --- факт3" (без кавычек)
            - Если коллизий нет: "" (пустая строка)

            Вопрос: {getData.question}

            Факты для анализа:
            {chr(10).join(f'- {fact}' for fact in getData.topFacts)}

            Ответ (строго в указанном формате):
            """

        try:
            response = self.llm.ask(prompt, specialization).strip()
            
            # Обработка ответа
            if response:  # Если есть непустой ответ
                try:
                    # Удаляем возможные кавычки и разбиваем по разделителю
                    cleaned_response = response.strip('"\'')
                    arrCollisionResult = [fact.strip() for fact in cleaned_response.split('---') if fact.strip()]
                    
                except Exception as e:
                    error.isError = True
                    error.messageError += f"Ошибка обработки ответа: {str(e)}"
            
            # Если ответ пустой или после обработки массив пуст - оставляем arrCollisionResult = []

        except Exception as e:
            error.isError = True
            error.messageError += f"Ошибка запроса к LLM: {str(e)}"

        return findCollisionsResult(arrCollisionResult=arrCollisionResult, error=error)


    # разбиение чанков на факты 
    def splittingChunksIntoFacts(self, getData: splittingChunksIntoFactsGet) -> splittingChunksIntoFactsResult:

        error = ErrorClass(False, "splittingChunksIntoFacts ")
        arrFacts = list()

        specialization = "Ты - эксперт по выделению фактов из текста."

        prompt = (f"""Выдели из текста факты

            Формат ответа (без кавычек): "факт_1 --- факт_2 --- ... --- факт_N"

            Текст: {getData.chunk}""")
        
        try:
            response = self.llm.ask(prompt, specialization)
            
            # Парсинг ответа с разделителем "---"
            try:
                # Разбиваем строку по разделителю "---"
                arrFacts = [fact.strip() for fact in response.split('---')]

            except Exception as e:
                error.isError = True
                error.messageError += f"Ошибка парсинга ответа: {str(e)}. Ответ LLM: {response}"
                
        except Exception as e:
            error.isError = True
            error.messageError += f"Ошибка при запросе к LLM: {str(e)}"

        return splittingChunksIntoFactsResult(arrFacts, error)
    


if __name__ == "__main__":
    service = LLMService()

    """
    # Пример использования findTopNearestLLM 
    test_data = findTopNearestLLMGet(
        question="Электромобили менее экологичны чем бензиновые",
        topFacts=[
            "Автомобили загрязняют воздух",
            "Бутерброды очень вкусные",
            "Электромобили более экологичны чем бензиновые",
            "Метро перевозит много людей с малым воздействием на экологию",
            "Самолеты производят много CO2"
        ],
        countFind=3
    )
    
    result = service.findTopNearestLLM(test_data)
    
    if result.error.isError:
        print(f"Ошибка: {result.error.messageError}")
    else:
        print("Найденные ближайшие факты:")
        for i, fact in enumerate(result.topNearest, 1):
            print(f"{i}. {fact}")
    """


    
    # Пример использования findCollisions
    """
    test_data = findCollisionsGet(
        question="Электромобили менее экологичны чем бензиновые",
        # question="Автомобили на электроэнергии более безопасны для окружающей среды чем бензиновые",
        # question="Электромобили менее экологичны чем бензиновые, а автомобили на электроэнергии более безопасны для окружающей среды те, что на бензине",
        #question="Доска в кабинете серая",
        topFacts=[
            "Автомобили загрязняют воздух",
            "Бутерброды очень вкусные",
            "Электромобили более экологичны чем бензиновые",
            "Метро перевозит много людей с малым воздействием на экологию",
            "Самолеты производят много CO2"
        ]
    )
    """

    test_data = findCollisionsGet(
        question="Собака сухая",
        topFacts=[
            "На улице дождь",
            "Собака гуляла на улице"
        ]
    )
    
    result = service.findCollisions(test_data)
    
    if result.error.isError:
        print(f"Ошибка: {result.error.messageError}")
    else:
        print("Ответ модели:")
        for i, ans in enumerate(result.arrCollisionResult, 1):
            print(f"{i}: {ans}")
    

    """
    # Пример использования splittingChunksIntoFacts    
    test_data = splittingChunksIntoFactsGet(
        chunk="в новой архитектурной мощи. Египетские пирамиды — древние монументальные сооружения, построенные египтянами для погребения фараонов и членов их семей. Преимущественно расположены" \
        " Преимущественно расположены на западном берегу реки Нил. Методы строительства пирамид до сих пор вызывают споры. Считается, что для " \
        "перемещения и подъёма блоков использовались рампы, рычаги и система примитивных катков. В 2017 году археологи обнаружили древний папирус," \
        " на котором был подробно описан план строительства пирамиды Хеопса."
    )
    
    result = service.splittingChunksIntoFacts(test_data)
    
    if result.error.isError:
        print(f"Ошибка: {result.error.messageError}")
    else:
        print("Найденные факты:")
        for i, fact in enumerate(result.arrFacts, 1):
            print(f"{i}. {fact}")
    """