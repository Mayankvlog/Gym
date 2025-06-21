import pandas as pd
import numpy as np
import pymongo
import uuid
import streamlit as st

# ================== Load Dataset ==================
df = pd.read_csv("megaGymDataset3.csv")[['Title', 'Desc']].dropna()
all_words = set(" ".join(df['Desc'].str.lower()).split())
vocab = {word: i for i, word in enumerate(all_words)}

# ================== BOW Embedder ==================
def embed(text):
    vec = np.zeros(len(vocab))
    for word in text.lower().split():
        if word in vocab:
            vec[vocab[word]] += 1
    return vec

# ================== Matrix Embedder ==================
def advanced_matrix_embed(query_vec, doc_texts):
    doc_matrix = np.array([embed(doc) for doc in doc_texts])
    query_mat = np.tile(query_vec, (doc_matrix.shape[0], 1))
    combined_matrix = np.hstack([
        query_mat,
        doc_matrix,
        query_mat * doc_matrix,
        np.abs(query_mat - doc_matrix)
    ])
    return combined_matrix

# ================== MongoDB Setup ==================
client = pymongo.MongoClient("mongodb+srv://mayankkr0311:mala@cluster0.yuxpwgl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
collection = client["MegaGymData"]["Gym"]

if collection.count_documents({}) == 0:
    for _, row in df.iterrows():
        collection.insert_one({
            "_id": str(uuid.uuid4()),
            "title": row['Title'],
            "desc": row['Desc'],
            "embedding": embed(row['Desc']).tolist()
        })

# ================== Neural Network Retriever ==================
class NeuralRetriever:
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64]):
        self.layer_dims = [input_dim] + hidden_dims + [1]
        self.weights = [
            np.random.randn(self.layer_dims[i], self.layer_dims[i+1]) *
            np.sqrt(2. / (self.layer_dims[i] + self.layer_dims[i+1])) 
            for i in range(len(self.layer_dims) - 1)
        ]
        self.biases = [np.zeros(self.layer_dims[i+1]) for i in range(len(self.layer_dims) - 1)]

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_activation(self, x):
        return np.maximum(0, x)
    
    def relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def relu_deriv_activation(self, x):
        return (x > 0).astype(float)
    
    def leaky_relu(self, x, alpha=0.01):    
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_activation(self, x, alpha=0.01):   
        return np.where(x > 0, x, alpha * x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_activation(self, x):   
        return np.tanh(x)
    
    def tanh_deriv(self, x):
        return 1.0 - np.tanh(x) ** 2        
    
    def tanh_deriv_activation(self, x):
        return 1.0 - np.tanh(x) ** 2

    def tanh_deriv_activation(self, x):
        return 1.0 - np.tanh(x) ** 2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-x)) 
    
    def softmax(self, x):    
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def softmax_activation(self, x):        
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def  cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-15
        return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))    
    
    def cross_entropy_loss_activation(self, y_true, y_pred):
        epsilon = 1e-15
        return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def mean_squared_error_activation(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)  


    def forward(self, x_batch):
        self.activations = [x_batch]
        self.z_values = []
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.relu(z))
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(self.sigmoid(z))
        return self.activations[-1]

    def backward(self, x_batch, y_true, y_pred, lr=0.01):
        m = y_true.shape[0]
        delta = 2 * (y_pred - y_true) * y_pred * (1 - y_pred) / m
        for i in reversed(range(len(self.weights))):
            a_prev = self.activations[i]
            dW = np.dot(a_prev.T, delta)
            db = np.sum(delta, axis=0)
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_deriv(self.z_values[i - 1])

    def score(self, query_vec, embeddings):
        scores = []
        for emb in embeddings:
            input_vec = np.hstack([
                query_vec,
                emb,
                query_vec * emb,
                np.abs(query_vec - emb)
            ])
            score = self.forward(input_vec.reshape(1, -1))
            scores.append(score[0])
        return scores

# ================== Generate Training Data ==================
def generate_training_data(docs, num_negatives=2):
    data = []
    for doc in docs:
        query = doc["desc"]
        data.append((query, doc["desc"], 1))  # Positive pair

        # Negative samples
        negatives = np.random.choice(
            [d for d in docs if d["_id"] != doc["_id"]],
            size=min(num_negatives, len(docs)-1),
            replace=False
        )
        for neg_doc in negatives:
            data.append((query, neg_doc["desc"], 0))  # Negative pair
    return data

# ================== Training Function ==================
def train_retriever(retriever, train_data, batch_size=2, epochs=10, lr=0.01):
    for epoch in range(epochs):
        np.random.shuffle(train_data)
        total_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            for query, doc, label in batch:
                query_vec = embed(query)
                doc_vec = embed(doc)
                input_vec = np.hstack([
                    query_vec,
                    doc_vec,
                    query_vec * doc_vec,
                    np.abs(query_vec - doc_vec)
                ])
                x_batch = np.array([input_vec])
                y_true = np.array([[label]])
                y_pred = retriever.forward(x_batch)
                retriever.backward(x_batch, y_true, y_pred, lr=lr)
                total_loss += np.mean((label - y_pred) ** 2)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ================== Train on Real Data ==================
docs = list(collection.find({}))
train_data = generate_training_data(docs)
retriever = NeuralRetriever(input_dim=len(vocab) * 4)
train_retriever(retriever, train_data)

# ================== Streamlit UI ==================
st.set_page_config(page_title="Gym AI Assistant", page_icon="🏋️")
st.title("🏋️‍♀️ Gym Workout Trainer")

query_input = st.text_input("Enter your fitness query:")

if st.button("Retrieve Best Match"):
    docs = list(collection.find({}))
    if len(docs) == 0:
        st.warning("No documents found in the database.")
    else:
        doc_texts = [doc['desc'] for doc in docs]
        query_vec = embed(query_input)
        combined_matrix = advanced_matrix_embed(query_vec, doc_texts)
        scores = retriever.forward(combined_matrix)
        top_idx = np.argmax(scores)
        top_doc = docs[top_idx]
        st.subheader("🏆 Top Result")
        st.write(f"**Title:** {top_doc['title']}")
        st.write(f"**Description:** {top_doc['desc']}")
