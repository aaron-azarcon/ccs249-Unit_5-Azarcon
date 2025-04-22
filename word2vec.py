import wikipedia
import re
import gensim
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# 1. I extended this part to include more topics and keywords
#    to ensure a more comprehensive dataset for training the model.
topics = {
    "Technological Singularity": ["Technological Singularity", "Future of AI", "Posthumanism"],
    "Artificial Intelligence": ["Artificial Intelligence", "AI development", "AI safety"],
    "Moor's Law": ["Transistors", "Gordon Moore", "Transistor scaling"],
    "Neural Networks": ["Neural Networks", "Deep Neural Networks", "Backpropagation"],
    "Quantum Computing": ["Quantum Computing", "Quantum Bits", "Quantum Supremacy"]
}

# 2. Fetch Wikipedia content
def fetch_content(topic_terms):
    content = ""
    for term in topic_terms:
        try:
            page = wikipedia.page(term)
            content += page.content[:2000] + " "  # More data helps Word2Vec
        except Exception as e:
            print(f"Error fetching '{term}': {e}")
    return content

# 3. Initial tokenized docs for Word2Vec training
tokenized_docs = []
for keywords in topics.values():
    combined = fetch_content(keywords)
    tokens = re.findall(r'\b\w+\b', combined.lower())
    tokenized_docs.append(tokens)

# 4. Train Word2Vec model
model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    seed=42
)

# 5. Create multiple samples (1 per keyword page)
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_doc_vector(tokens, model):
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

X = []
y = []
for i, (topic_name, keywords) in enumerate(topics.items()):
    for term in keywords:
        content = fetch_content([term])
        tokens = tokenize(content)
        vector = get_doc_vector(tokens, model)
        if np.any(vector):
            X.append(vector)
            y.append(i)

X = np.array(X)
y = np.array(y)

# 6. Train Logistic Regression classifier
clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf.fit(X, y)

# 7. Predict and evaluate
y_pred = clf.predict(X)
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=list(topics.keys()), zero_division=0))
