import wikipedia
import re
import math
from collections import Counter
from math import log
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# makes use of the same code from number 1 but with a few changes for the cosine similarity

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
        return page.content[:1000]
    except Exception as e:
        print(f"Error fetching {topic}: {e}")
        return ""

def tokenize(text):
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in stop_words]

def compute_tf(tokens, vocab):
    count = Counter(tokens)
    total_terms = len(tokens)
    return { term: count[term] / total_terms for term in vocab }

def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        df = sum(term in doc for doc in tokenized_docs)
        idf_dict[term] = log(N / (df or 1))
    return idf_dict

def compute_tfidf(tf_vector, idf, vocab):
    return { term: tf_vector[term] * idf[term] for term in vocab }

def cosine_similarity(vec1, vec2, vocab):
    dot_product = sum(vec1[term] * vec2[term] for term in vocab)
    vec1_len = math.sqrt(sum(vec1[term]**2 for term in vocab))
    vec2_len = math.sqrt(sum(vec2[term]**2 for term in vocab))

    if vec1_len == 0 or vec2_len == 0:
        return 0.0
    return dot_product / (vec1_len * vec2_len)

# === Main Execution ===
docs = {topic: fetch_content(topic) for topic in topics}
tokenized_docs = [tokenize(doc) for doc in docs.values()]
vocab = sorted(set(word for doc in tokenized_docs for word in doc))

tf_vectors = [compute_tf(tokens, vocab) for tokens in tokenized_docs]
idf = compute_idf(tokenized_docs, vocab)
tfidf_vectors = [compute_tfidf(tf, idf, vocab) for tf in tf_vectors]

doc1 = "Technological Singularity"
doc2 = "Artificial Intelligence"

index1 = topics.index(doc1)
index2 = topics.index(doc2)

similarity = cosine_similarity(tfidf_vectors[index1], tfidf_vectors[index2], vocab)
print(f"Cosine similarity between '{doc1}' and '{doc2}': {similarity:.4f}")
