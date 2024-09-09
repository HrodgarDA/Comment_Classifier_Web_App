import re
import nltk
from nltk.corpus import stopwords
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

def load_resources():
    with open('../model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('../model/config.json', 'r') as f:
        config = json.load(f)
    return tokenizer, config

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def prepare_input(text, tokenizer, config):
    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=config['MAX_LENGTH'])
    return padded_sequence

def predict_toxicity(text): #Prediction function
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH)
    prediction = model.predict(padded)
    binary_prediction = (prediction > 0.5).astype(int)[0]
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] # Lista delle classi

    positive_classes = [classes[i] for i, value in enumerate(binary_prediction) if value == 1]

    if positive_classes == []:
      print("The text sample provided is approved")
    else:
        print(f"The text sample provided is classified as: {positive_classes}")

    return binary_prediction, positive_classes




