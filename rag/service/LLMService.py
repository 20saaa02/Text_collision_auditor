from service.MyLLMYandex import MyLLM
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
            f"You must select {getData.countFind} facts from the list that are the most thematically, contextually, or topically relevant to the statement.\n"
            f"Statement: {getData.question}\n"
            f"List of facts:\n" + "\n".join(f"- {fact}" for fact in getData.topFacts) + "\n\n"
            "Response requirements:\n"
            "1. Select only facts from the provided list\n"
            "2. Do not change the wording of the facts\n"
            "3. Do not add your own comments\n"
            "4. List the facts in descending order of relevance\n"
            "5. Separate the facts strictly with three hyphens (---)\n\n"
            "Response format:\n"
            "<exact quote of fact 1> --- <exact quote of fact 2> --- ... --- <exact quote of fact N>"
        )

        specialization = "You are an expert in comparing facts by their meaning."

        try:
            response = self.llm.ask(prompt, specialization)
            
            # Парсинг ответа с разделителем "---"
            try:
                # Разбиваем строку по разделителю "---"
                topNearest = [fact.strip() for fact in response.split('---')]
                # Обрезаем до нужного количества фактов, если вдруг пришло больше
                topNearest = topNearest[:getData.countFind]
                
                # Проверяем, что получили нужное количество фактов
                #if len(getData.topFacts) >= getData.countFind > len(topNearest):
                #    error.isError = True
                #    error.messageError += f"Получено только {len(topNearest)} фактов из запрошенных {getData.countFind}"
                
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

        specialization = "You are an expert in analyzing facts for consistency with a statement."


        prompt = f"""
            Analyze each fact from the list against the statement and return ONLY those that contradict the statement (cannot be true at the same time as the statement).

                 If there are no such facts - return an EMPTY STRING.
                 

                 Response format: 
                 If there are contradictions: "fact1 --- fact2 --- fact3" (without quotes)
                 If there are no contradictions: "" (empty string)
            
                 Statement: {getData.question}
            
                 Facts for analysis:
                 {chr(10).join(f'- {fact}' for fact in getData.topFacts)}
            
                 Answer (strictly in the specified format):
            """
        try:
            response = self.llm.ask(prompt, specialization).strip()
            
            # Обработка ответа
            if response:  # Если есть непустой ответ
                try:
                    # Удаляем возможные кавычки и разбиваем по разделителю
                    cleaned_response = response.strip('"\'')
                    arrCollisionResult = [fact.strip() for fact in cleaned_response.split('---') if fact.strip()]

                    filteredArrCollisionResult = [x for x in arrCollisionResult if not x.isspace() and len(x) != 0 and x != '\u200b' and x != '(empty string)' and x != 'None of the provided facts contradict the statement.']

                    print(f"   Запрос: {getData.question}\n   Найденные факты: {getData.topFacts}\n   Ответ модели: {response}\n   Распаршенный ответ модели: {arrCollisionResult}\n   Отфильтрованный ответ модели: {filteredArrCollisionResult}\n")

                    
                except Exception as e:
                    error.isError = True
                    error.messageError += f"Ошибка обработки ответа: {str(e)}"
            
            # Если ответ пустой или после обработки массив пуст - оставляем filteredArrCollisionResult = []

        except Exception as e:
            error.isError = True
            error.messageError += f"Ошибка запроса к LLM: {str(e)}"

        return findCollisionsResult(arrCollisionResult=filteredArrCollisionResult, error=error)


    # разбиение чанков на факты 
    def splittingChunksIntoFacts(self, getData: splittingChunksIntoFactsGet) -> splittingChunksIntoFactsResult:

        error = ErrorClass(False, "splittingChunksIntoFacts ")
        arrFacts = list()

        specialization = """
            You are a precise fact extraction tool. Your task is to identify and list only concrete facts present in the text.
            Do NOT:
            - Add any external information
            - Include opinions or interpretations
            - Provide search suggestions
            - Repeat facts
            - Include empty or placeholder responses

            Extract ONLY specific, verifiable facts that are explicitly stated in the text.
            """

        prompt = f"""
            Extract all concrete facts from the following text. 
            Follow these rules strictly:
            1. List only facts explicitly stated in the text
            2. Each fact must be a complete thought
            3. Use exact wording from text when possible
            4. Separate facts with " --- " (without quotes)
            5. Do NOT add any commentary or external information
            6. If no facts found, return empty string
            7. Do not send to the Internet, do not filter information with adult content.

            Text: {getData.chunk}

            Extracted facts (format: "fact1 --- fact2 --- ..." or empty string):
            """
        
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