import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
#from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # или WordNetLemmatizer
from entity.dataPreproc import ErrorClass, cleanTextGet, cleanTextResult, splittingTextIntoChunksGet, splittingTextIntoChunksResult, initPreprocessingDataServiceGet
    
# предобработка текста (лучше делить на чанки перед очисткой текста!)
class PreprocessingDataService:
    def __init__(self, _: initPreprocessingDataServiceGet):
        #self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()  # Альтернатива: WordNetLemmatizer()

    # очистка текста
    def cleanText(self, getData: cleanTextGet) -> cleanTextResult:
        """
            Полная очистка английского текста:
            1. Lowercasing
            2. Удаление пунктуации
            3. Удаление стоп-слов
            4. Стемминг
        """
        text = getData.text
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove punctuation
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # 3. Tokenization
        tokens = word_tokenize(text)
        
        # 4. Processing
        cleaned_tokens = []
        for token in tokens:
            #if token in self.stop_words:
            #    continue
                
            token = self.stemmer.stem(token)
                
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
                # Разбиваем по предложениям с сохранением контекста
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
                        
                        # Добавляем перекрытие
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
                # Простое разбиение по фиксированной длине
                for i in range(0, len(text), getData.chunk_size - getData.overlap):
                    chunk = text[i:i + getData.chunk_size]
                    chunks.append(chunk)
                    
        except Exception as e:
            error.isError = True
            error.messageError = f"Ошибка при разбиении текста: {str(e)}"
        
        return splittingTextIntoChunksResult(chunks=chunks, error=error)

        
if __name__ == "__main__":
    service = PreprocessingDataService()

    # Тест очистки текста
    test_clean = cleanTextGet(
        text="The quick brown fox jumps over the lazy dog 123! "
        "Natural Language Processing (NLP) is amazing!!! I'd rather have a coffee at 3pm, wouldn't you?"
    )
    clean_result = service.cleanText(test_clean)
    print(f"Очищенный текст:\n{clean_result.text}\n")

    # Тест разбиения на чанки
    test_chunk = splittingTextIntoChunksGet(
        text="Large language models (LLMs) have revolutionized NLP. They can generate human-like text. "
        "However, LLMs have limitations. They may produce incorrect information. "
        "This is known as the 'hallucination' problem. RAG combines retrieval and generation. "
        "It first retrieves relevant documents, then generates answers based on them.",
        chunk_size=100,
        overlap=30,
        by_sentences=True
    )
    chunk_result = service.splittingTextIntoChunks(test_chunk)
    
    print(f"Разбиение на чанки (ошибка: {chunk_result.error.isError}):")
    for i, chunk in enumerate(chunk_result.chunks, 1):
        print(f"\nЧанк {i} ({len(chunk)} chars):\n{chunk}")