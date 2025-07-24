# LLMService.py
from concurrent.futures import ThreadPoolExecutor
from service.MyLLMYandex import MyLLM
from entity.dataLLM import ErrorClass, findTopNearestLLMGet, findTopNearestLLMResult, findCollisionsResult, \
    findCollisionsGet, \
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

            try:
                topNearest = [fact.strip() for fact in response.split('---')]
                topNearest = topNearest[:getData.countFind]

                if len(getData.topFacts) >= getData.countFind > len(topNearest):
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

            if response:
                try:
                    cleaned_response = response.strip('"\'')
                    arrCollisionResult = [fact.strip() for fact in cleaned_response.split('---') if fact.strip()]
                    filteredArrCollisionResult = [x for x in arrCollisionResult if
                                                  x in getData.topFacts and len(x) != 0]
                except Exception as e:
                    error.isError = True
                    error.messageError += f"Ошибка обработки ответа: {str(e)}"
        except Exception as e:
            error.isError = True
            error.messageError += f"Ошибка запроса к LLM: {str(e)}"

        return findCollisionsResult(arrCollisionResult=filteredArrCollisionResult, error=error)

    # разбиение чанков на факты
    def splittingChunksIntoFacts(self, getData: splittingChunksIntoFactsGet) -> splittingChunksIntoFactsResult:

        error = ErrorClass(False, "splittingChunksIntoFacts ")
        arrFacts = list()
        specialization = "You are an expert in extracting facts from text."
        prompt = (f"""Extract facts from the text

            Response format (without quotes): "fact_1 --- fact_2 --- ... --- fact_N"

            Text: {getData.chunk}""")

        try:
            response = self.llm.ask(prompt, specialization)
            try:
                arrFacts = [fact.strip() for fact in response.split('---')]
            except Exception as e:
                error.isError = True
                error.messageError += f"Ошибка парсинга ответа: {str(e)}. Ответ LLM: {response}"
        except Exception as e:
            error.isError = True
            error.messageError += f"Ошибка при запросе к LLM: {str(e)}"

        return splittingChunksIntoFactsResult(arrFacts, error)

    def splittingChunksIntoFacts_parallel(self, list_of_get_data: list[splittingChunksIntoFactsGet]) -> list[
        splittingChunksIntoFactsResult]:
        """Распараллеливает запросы к LLM для извлечения фактов с помощью потоков."""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.splittingChunksIntoFacts, list_of_get_data))
        return results