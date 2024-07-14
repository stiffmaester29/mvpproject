import psycopg2
import re
import torch
from transformers import BertTokenizer, BertModel
from multiprocessing import Pool
from functools import partial

# Connection parameters
conn_params = {
    "dbname": "mvp",
    "user": "postgres",
    "password": "larry",
    "host": "localhost",
    "port": "5432"
}


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def get_data(table_name):
    """ Query data from the specified table """
    sql = f"SELECT prod_desc FROM {table_name}"
    conn = None
    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return [row[0] for row in rows]  # Return only the first element of each tuple
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text


def get_bert_embeddings(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Get the mean of the embeddings


def compute_similarity(test_item, control_items, model, tokenizer):
    similarities = []
    preprocessed_test = preprocess(test_item)

    test_embeddings = get_bert_embeddings([preprocessed_test], model, tokenizer)[0]

    for control_item in control_items:
        preprocessed_control = preprocess(control_item)
        control_embeddings = get_bert_embeddings([preprocessed_control], model, tokenizer)[0]
        similarity_score = torch.cosine_similarity(control_embeddings, test_embeddings, dim=0).item() * 100  # Get similarity as percentage

        similarities.append({
            'control': preprocessed_control,
            'test': preprocessed_test,
            'similarity': similarity_score
        })
    return similarities


if __name__ == '__main__':
    connect()

    control = get_data('tbl_control')
    test = get_data('tbl_test')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    with Pool() as pool:
        compute_similarity_partial = partial(compute_similarity, control_items=control, model=model, tokenizer=tokenizer)
        all_similarities = pool.map(compute_similarity_partial, test)

    # Flatten the list of similarities
    similarities = [item for sublist in all_similarities for item in sublist]

    # Use a set to track unique combinations of control and test
    unique_similarities = {(item['control'], item['test']): item for item in similarities}

    # Convert the unique similarities back to a list
    unique_similarities_list = list(unique_similarities.values())

    # Sort the unique similarities in descending order and get the top 250
    top_similarities = sorted(unique_similarities_list, key=lambda x: x['similarity'], reverse=True)[:250]

    # Print the top similarities
    print("\nTop 3 Highest Similarities:")
    for item in top_similarities[:3]:  # Print only the top 3
        print(f"Control: {item['control']}, Test: {item['test']}, Similarity: {item['similarity']:.2f}%")
