import wikipedia
import re
import math
from collections import Counter
from math import log
from prettytable import PrettyTable
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')  # for removing stopwords

topics = [
    "Technological Singularity",
    "Artificial Intelligence",
    "Moor's Law",
    "Neural Networks",
    "Quantum Computing"
]

def fetch_content(topic):
    try:
        page = wikipedia.page(topic)
        return page.content[:1000]  # Limit to 1000 characters for simplicity
    except Exception as e:
        print(f"Error fetching {topic}: {e}")
        return ""

docs = {topic: fetch_content(topic) for topic in topics}

stop_words = set(stopwords.words('english'))

def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in stop_words]

tokenized_docs = [tokenize(doc) for doc in docs.values()]
vocab = sorted(set(word for doc in tokenized_docs for word in doc))

def compute_tf(tokens, vocab):
    count = Counter(tokens)
    total_terms = len(tokens)
    return { term: count[term] / total_terms for term in vocab }

tf_vectors = [compute_tf(tokens, vocab) for tokens in tokenized_docs]

def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        df = sum(term in doc for doc in tokenized_docs)
        idf_dict[term] = log(N / (df or 1))
    return idf_dict

idf = compute_idf(tokenized_docs, vocab)

def compute_tfidf(tf_vector, idf, vocab):
    return { term: tf_vector[term] * idf[term] for term in vocab }

tfidf_vectors = [compute_tfidf(tf, idf, vocab) for tf in tf_vectors]

def build_table(matrix, vocab, doc_names, title):
    table = PrettyTable()
    table.field_names = ["Term"] + doc_names

    for term in vocab:
        row = [term]
        for vec in matrix:
            val = vec.get(term, 0)
            row.append(round(val, 4) if val > 0 else "")
        if any(row[1:]):  # Only include rows with at least one non-zero
            table.add_row(row)
    table.title = title
    return table

doc_names = list(docs.keys())

# Raw Frequency Table
print(build_table(tf_vectors, vocab, doc_names, "Raw Term Frequency Matrix"))

# TF-IDF Table
print(build_table(tfidf_vectors, vocab, doc_names, "TF-IDF Matrix"))
