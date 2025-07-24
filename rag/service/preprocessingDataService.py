# preprocessingDataService.py
import re
import string
from concurrent.futures import ProcessPoolExecutor
from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from entity.dataPreproc import ErrorClass, cleanTextGet, cleanTextResult, splittingTextIntoChunksGet, \
    splittingTextIntoChunksResult, initPreprocessingDataServiceGet


# --- WORKERS ДЛЯ МУЛЬТИПРОЦЕССИНГА ---

# def _worker_split_text(args):
#     """Вспомогательная функция для параллельного вызова splittingTextIntoChunks."""
#     service_instance, get_data = args
#     return service_instance.splittingTextIntoChunks(get_data)
#
#
# def _worker_clean_text(args):
#     """Вспомогательная функция для параллельного вызова cleanText."""
#     service_instance, get_data = args
#     return service_instance.cleanText(get_data)


# предобработка текста
class PreprocessingDataService:
    def __init__(self, _: initPreprocessingDataServiceGet):
        # self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    # очистка текста
    def cleanText(self, getData: cleanTextGet) -> cleanTextResult:
        text = getData.text
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        tokens = word_tokenize(text)
        cleaned_tokens = []
        for token in tokens:
            # if token in self.stop_words:
            #    continue
            # token = self.stemmer.stem(token)
            cleaned_tokens.append(token)

        return cleanTextResult(text=' '.join(cleaned_tokens).strip())

    # деление текста на чанки с сохранением смысловых границ
    def splittingTextIntoChunks(self, getData: splittingTextIntoChunksGet) -> splittingTextIntoChunksResult:
        error = ErrorClass(False, "")
        chunks = []

        try:
            text = getData.text.strip()
            if not text:
                return splittingTextIntoChunksResult(chunks=[], error=error)

            if getData.by_sentences:
                sentences = sent_tokenize(text)
                current_chunk = []
                current_length = 0

                for sentence in sentences:
                    sentence_length = len(sentence)

                    if current_length + sentence_length <= getData.chunk_size:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                    else:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))

                        overlap_chunk = []
                        overlap_length = 0
                        for s in reversed(current_chunk):
                            if overlap_length + len(s) <= getData.overlap:
                                overlap_chunk.insert(0, s)
                                overlap_length += len(s)
                            else:
                                break

                        current_chunk = overlap_chunk + [sentence]
                        current_length = overlap_length + sentence_length

                if current_chunk:
                    chunks.append(' '.join(current_chunk))
            else:
                for i in range(0, len(text), getData.chunk_size - getData.overlap):
                    chunk = text[i:i + getData.chunk_size]
                    chunks.append(chunk)

        except Exception as e:
            error.isError = True
            error.messageError = f"Ошибка при разбиении текста: {str(e)}"

        return splittingTextIntoChunksResult(chunks=chunks, error=error)

    # # --- ПАРАЛЛЕЛЬНЫЕ МЕТОДЫ ---
    #
    # def splittingTextIntoChunks_parallel(self, list_of_get_data: list[splittingTextIntoChunksGet]) -> list[
    #     splittingTextIntoChunksResult]:
    #     """Распараллеливает обработку списка текстов по разным процессам."""
    #     with ProcessPoolExecutor() as executor:
    #         args_for_workers = [(self, get_data) for get_data in list_of_get_data]
    #         results = list(executor.map(_worker_split_text, args_for_workers))
    #     return results
    #
    # def cleanText_parallel(self, list_of_get_data: list[cleanTextGet]) -> list[cleanTextResult]:
    #     """Распараллеливает очистку списка текстов (чанков) по разным процессам."""
    #     with ProcessPoolExecutor() as executor:
    #         args_for_workers = [(self, get_data) for get_data in list_of_get_data]
    #         results = list(executor.map(_worker_clean_text, args_for_workers))
    #     return results