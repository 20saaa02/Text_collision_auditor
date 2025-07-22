from MyLLM import MyLLM
from rag.entity.dataLLM import ErrorClass, findTopNearestGet, findTopNearestResult, findCollisionsResult, findCollisionsGet, CollisionAnswer, \
    findCollisionsResultOne, splittingChunksIntoFactsGet, splittingChunksIntoFactsResult
    

class LLMService:
    def __init__(self):
        self.llm = MyLLM()

    # выделить из топ-N топ-K ближайших по смыслу к question
    def findTopNearest(self, getData: findTopNearestGet) -> findTopNearestResult:

        error = ErrorClass(False, "")
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
                    error.messageError = f"Получено только {len(topNearest)} фактов из запрошенных {getData.countFind}"
                
            except Exception as e:
                error.isError = True
                error.messageError = f"Ошибка парсинга ответа: {str(e)}. Ответ LLM: {response}"
                
        except Exception as e:
            error.isError = True
            error.messageError = f"Ошибка при запросе к LLM: {str(e)}"

        return findTopNearestResult(topNearest, error)
    

    # функция поиска противоречий вопроса с топ-K сообщений
    def findCollisions(self, getData: findCollisionsGet) -> findCollisionsResult:
        error = ErrorClass(False, "")
        arrCollisionResult = []

        specialization = "Ты - эксперт по анализу фактов на предмет соответствия вопросу."

        prompt = f"""
            Проанализируй каждый факт из списка относительно вопроса и определи тип соответствия по следующим правилам:

            1. Если факт ПОДТВЕРЖДАЕТ вопрос (согласуется с ним по смыслу) → {CollisionAnswer.SUPPORT.name}
            2. Если факт НЕ ПРОТИВОРЕЧИТ вопросу (но и не подтверждает) → {CollisionAnswer.NO_COLLISION.name}
            3. Если факт ПРОТИВОРЕЧИТ вопросу (не может быть истинным одновременно) → {CollisionAnswer.COLLISION_BETWEEN_FACTS.name}

            Формат ответа: ТОЛЬКО список строк в формате:
            <факт> || <тип коллизии>

            Пример ответа:
            Факт 1 || SUPPORT
            Факт 2 || NO_COLLISION
            Факт 3 || COLLISION_BETWEEN_FACTS

            Если вопрос содержит внутреннее противоречие, верни:
            <вопрос> || {CollisionAnswer.COLLISION_IN_QUESTION.name}

            Вопрос для анализа: {getData.question}

            Факты для анализа:
            {chr(10).join(f'- {fact}' for fact in getData.topFacts)}

            Ответ (только в указанном формате, без дополнительных комментариев):
            """

        try:
            response = self.llm.ask(prompt, specialization)
            
            # Парсинг ответа
            for line in response.split('\n'):
                line = line.strip()
                if not line or '||' not in line:
                    continue
                    
                fact_part, collision_part = line.split('||', 1)
                fact = fact_part.strip()
                collision_type = collision_part.strip()
                
                try:
                    collision_enum = CollisionAnswer[collision_type]
                    arrCollisionResult.append(
                        findCollisionsResultOne(fact=fact, collisionType=collision_enum)
                    )
                except KeyError:
                    error.isError = True
                    error.messageError = "Модель вернула неизвестный тип."
                    break
                    
            # Если не найдено ни одного результата
            if not arrCollisionResult:
                error.isError = True
                error.messageError = "Модель не вернула ни одного корректного результата"
                # Возвращаем NO_COLLISION для всех фактов
                arrCollisionResult = [
                    findCollisionsResultOne(fact=fact, collisionType=CollisionAnswer.NO_COLLISION)
                    for fact in getData.topFacts
                ]
                
        except Exception as e:
            error.isError = True
            error.messageError = f"Ошибка при запросе к LLM: {str(e)}"

        return findCollisionsResult(arrCollisionResult=arrCollisionResult, error=error)


    # разбиение чанков на факты 
    def splittingChunksIntoFacts(self, getData: splittingChunksIntoFactsGet) -> splittingChunksIntoFactsResult:

        error = ErrorClass(False, "")
        arrFacts = list()

        specialization = "Ты - эксперт по выделению фактов из текста."

        prompt = (f"""Выдели из текста факты

            Формат ответа (без кавычек): "<факт 1> --- <факт 2> --- ... --- <факт N>"

            Текст: {getData.chunk}""")
        
        try:
            response = self.llm.ask(prompt, specialization)
            
            # Парсинг ответа с разделителем "---"
            try:
                # Разбиваем строку по разделителю "---"
                arrFacts = [fact.strip() for fact in response.split('---')]

            except Exception as e:
                error.isError = True
                error.messageError = f"Ошибка парсинга ответа: {str(e)}. Ответ LLM: {response}"
                
        except Exception as e:
            error.isError = True
            error.messageError = f"Ошибка при запросе к LLM: {str(e)}"

        return splittingChunksIntoFactsResult(arrFacts, error)
    


if __name__ == "__main__":
    service = LLMService()

    """
    # Пример использования findTopNearest    
    test_data = findTopNearestGet(
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
    
    result = service.findTopNearest(test_data)
    
    if result.error.isError:
        print(f"Ошибка: {result.error.messageError}")
    else:
        print("Найденные ближайшие факты:")
        for i, fact in enumerate(result.topNearest, 1):
            print(f"{i}. {fact}")
    """


    """
    # Пример использования findCollisions
    test_data = findCollisionsGet(
        # question="Электромобили менее экологичны чем бензиновые",
        # question="Автомобили на электроэнергии более безопасны для окружающей среды чем бензиновые",
        question="Электромобили менее экологичны чем бензиновые, а автомобили на электроэнергии более безопасны для окружающей среды те, что на бензине",
        # question="Доска в кабинете серая",
        topFacts=[
            "Автомобили загрязняют воздух",
            "Бутерброды очень вкусные",
            "Электромобили более экологичны чем бензиновые",
            "Метро перевозит много людей с малым воздействием на экологию",
            "Самолеты производят много CO2"
        ]
    )
    
    result = service.findCollisions(test_data)
    
    if result.error.isError:
        print(f"Ошибка: {result.error.messageError}")
    else:
        print("Ответ модели:")
        for ans in result.arrCollisionResult:
            print(ans.fact, ans.collisionType.name)
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