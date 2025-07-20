import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from pymongo import MongoClient

# ========== MongoDB Atlas Setup ==========
MONGODB_URI = "mongodb+srv://mayankkr0311:mala@cluster0.yuxpwgl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGODB_URI)
db = client['MegaGymData']
collection = db['user_queries']

def save_to_mongodb(title, context, generated_desc):
    doc = {
        'title': title,
        'context': context,
        'generated_description': generated_desc
    }
    collection.insert_one(doc)

# ========== Load and Prepare Data ==========
@st.cache_data(show_spinner=False)
def load_and_prepare():
    df = pd.read_csv('megaGymDataset3.csv')
    df = df.dropna(subset=['Title', 'Desc'])
    df['Desc'] = 'startseq ' + df['Desc'].astype(str) + ' endseq'
    df['context'] = df['Type'].fillna('') + ' ' + df['BodyPart'].fillna('') + ' ' + df['Equipment'].fillna('') + ' ' + df['Level'].fillna('')
    return df

df = load_and_prepare()

# ========== Retriever ==========
vectorizer = TfidfVectorizer(max_features=500)
tfidf_matrix = vectorizer.fit_transform((df['Title'] + ' ' + df['context']).astype(str))
retriever = NearestNeighbors(n_neighbors=1, metric='cosine').fit(tfidf_matrix)

def retrieve_similar_desc(title, context):
    query = title + ' ' + context
    X_query = vectorizer.transform([query])
    idx = retriever.kneighbors(X_query, return_distance=False)[0][0]
    return df.iloc[idx]['Desc']

# ========== Tokenization ==========
input_tokenizer = Tokenizer(oov_token='<OOV>')
desc_tokenizer = Tokenizer(oov_token='<OOV>')

df['encoder_input'] = df['Title'].astype(str) + ' ' + df['Desc'].astype(str)
input_tokenizer.fit_on_texts(df['encoder_input'])
desc_tokenizer.fit_on_texts(df['Desc'])

max_input_len = 60
max_desc_len = 50
input_vocab_size = len(input_tokenizer.word_index) + 1
desc_vocab_size = len(desc_tokenizer.word_index) + 1

encoder_input_seqs = input_tokenizer.texts_to_sequences(df['encoder_input'])
desc_seqs = desc_tokenizer.texts_to_sequences(df['Desc'])
encoder_input_seqs = pad_sequences(encoder_input_seqs, maxlen=max_input_len, padding='post')
desc_seqs = pad_sequences(desc_seqs, maxlen=max_desc_len, padding='post')

# ========== Model Definition ==========
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    try:
        model = load_model('gym.h5')
        with open('input_tokenizer.pkl', 'rb') as f:
            input_tok = pickle.load(f)

        with open('desc_tokenizer.pkl', 'rb') as f:
            desc_tok = pickle.load(f)
        return model, input_tok, desc_tok
    except Exception:

        # Train model if not found
        X_enc_train, X_enc_test, y_desc_train, y_desc_test = train_test_split(
            encoder_input_seqs, desc_seqs, test_size=0.1, random_state=42
        )
        encoder_input_layer = Input(shape=(max_input_len,), name='encoder_input')
        x = Embedding(input_dim=input_vocab_size, output_dim=128, mask_zero=True)(encoder_input_layer)
        x = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(x)
        x = LSTM(64, return_sequences=True, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = LSTM(32, return_sequences=True, activation='sigmoid')(x)
        x = LSTM(32, return_sequences=True, activation='elu')(x)
        encoder_output, state_h, state_c = LSTM(32, return_state=True, activation='relu')(x)

        decoder_input = Input(shape=(max_desc_len,), name='decoder_input')
        decoder_embedding = Embedding(input_dim=desc_vocab_size, output_dim=128, mask_zero=True)(decoder_input)
        y = LSTM(32, return_sequences=True, activation='tanh')(decoder_embedding, initial_state=[state_h, state_c])
        y = Dropout(0.2)(y)
        y = LSTM(32, return_sequences=True, activation='relu')(y)
        output = Dense(desc_vocab_size, activation='softmax')(y)

        model = Model([encoder_input_layer, decoder_input], output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        decoder_input_data = np.zeros_like(y_desc_train)
        decoder_input_data[:, 1:] = y_desc_train[:, :-1]
        decoder_input_data[:, 0] = desc_tokenizer.word_index.get('startseq', 0)
        decoder_target_data = tf.keras.utils.to_categorical(y_desc_train, num_classes=desc_vocab_size)

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit([X_enc_train, decoder_input_data], decoder_target_data,
                            batch_size=32, epochs=300, validation_split=0.1, callbacks=[early_stop])
        model.save('gym.h5')
        with open('input_tokenizer.pkl', 'wb') as f:
            pickle.dump(input_tokenizer, f)
        with open('desc_tokenizer.pkl', 'wb') as f:
            pickle.dump(desc_tokenizer, f)
        return model, input_tokenizer, desc_tokenizer

model, input_tokenizer, desc_tokenizer = load_or_train_model()

# ========== Inference ==========
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_description(input_title, input_context, max_gen_len=50, temperature=0.8):
    retrieved_desc = retrieve_similar_desc(input_title, input_context)
    encoder_input_text = input_title + ' ' + retrieved_desc
    encoder_input_seq = input_tokenizer.texts_to_sequences([encoder_input_text])
    encoder_input_seq = pad_sequences(encoder_input_seq, maxlen=max_input_len, padding='post')
    decoder_input_inf = np.zeros((1, max_desc_len))
    decoder_input_inf[0, 0] = desc_tokenizer.word_index.get('startseq', 0)
    generated_sequence = []
    for i in range(1, max_gen_len):
        predictions = model.predict([encoder_input_seq, decoder_input_inf], verbose=0)
        preds = predictions[0, i-1, :]
        predicted_word_index = sample_with_temperature(preds, temperature)
        if predicted_word_index == desc_tokenizer.word_index.get('endseq', -1) or predicted_word_index == 0:
            break
        generated_sequence.append(predicted_word_index)
        if i < max_desc_len:
            decoder_input_inf[0, i] = predicted_word_index
    inv_desc_index = {v: k for k, v in desc_tokenizer.word_index.items()}
    generated_desc = ' '.join([inv_desc_index.get(idx, '') for idx in generated_sequence if idx > 0])
    return generated_desc

# ========== Streamlit UI ==========
st.title("🏋️‍♂️ Gym Exercise Assistant")

with st.form("desc_form"):
    title_input = st.text_input("Enter Exercise Title", value="abs exercise in 3 days")
    context_input = st.text_input("Enter Context (Type, BodyPart, Equipment, Level)", value="Strength abs Dumbbell advanced")
    submitted = st.form_submit_button("Generate Description")

if submitted:
    if title_input and context_input:
        with st.spinner("Generating..."):
            generated_desc = generate_description(title_input, context_input)
            st.markdown("**Generated Description:**")
            st.write(generated_desc)
            save_to_mongodb(title_input, context_input, generated_desc)
            st.success("Description saved to MongoDB Atlas.")
    else:
        st.warning("Please provide both title and context.")


