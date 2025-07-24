import pandas as pd

# data_train = pd.read_csv(r"C:\Users\memel\PycharmProjects\Text_collision_auditor\DATA\data_train.csv")
data_test = pd.read_csv(r"C:\Users\memel\PycharmProjects\DATA\data_test.csv")
corpus = pd.read_json(r"C:\Users\memel\PycharmProjects\DATA\list_corpus.jsonl", lines=True)
queries = pd.read_json(r"C:\Users\memel\PycharmProjects\DATA\list_queries.jsonl", lines=True)

# Создаем таблицу texts
texts = []

# Получаем уникальные пары (queries_id, corpus_id)
unique_pairs = data_test[['queries_id', 'corpus_id']].drop_duplicates()

text_id_counter = 0

for _, row in unique_pairs.iterrows():
    q_id = row['queries_id']
    c_id = row['corpus_id']

    # Получаем claim
    claim_row = queries[queries['id'] == q_id]
    if claim_row.empty:
        continue
    claim = claim_row.iloc[0]['claim']

    # Получаем abstract
    corpus_row = corpus[corpus['doc_id'] == c_id]
    if corpus_row.empty:
        continue
    abstract = corpus_row.iloc[0]['abstract']

    # Собираем text = abstract + [claim]
    text = abstract + [claim]
    len_text = len(text)

    # Собираем CONTRADICT пары
    contradict_rows = data_test[
        (data_test['queries_id'] == q_id) &
        (data_test['corpus_id'] == c_id) &
        (data_test['label'] == 'CONTRADICT')
    ]

    # Формируем true_pair: список [(len_text - 1, sentence_id)]
    true_pair = [(len_text - 1, sid) for sid in contradict_rows['sentences_id'].tolist()]

    # Добавляем запись в таблицу texts
    texts.append({
        'text_id': text_id_counter,
        'text': text,
        'len_text': len_text,
        'true_pair': true_pair
        # 'predict_pair': можно оставить пустым или добавить позже
    })

    text_id_counter += 1

# Финальный DataFrame
texts_train_df = pd.DataFrame(texts).set_index("text_id")
texts_train_df.to_csv(r"C:\Users\memel\PycharmProjects\Text_collision_auditor1\rag\measurements\data_test_for_measure.csv")
print(texts_train_df.head())
