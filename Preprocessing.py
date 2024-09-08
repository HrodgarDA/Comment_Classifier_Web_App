import os
import pickle
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Comments_Preprocessing import preprocess_text

# Definisci il percorso della directory dove sono salvati i file del modello
model_dir = '/Users/rugg/Documents/GitHub/Comment_Classifier_Web_App/toxic_comment_model.h5'

# Carica il modello
model = tf.keras.models.load_model(os.path.join(model_dir, 'RNN_model.h5'))

# Carica il tokenizer
with open(os.path.join(model_dir, 'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)

# Carica la configurazione
with open(os.path.join(model_dir, 'model_parameters.json'), 'r') as f:
    config = json.load(f)

MAX_LENGTH = config['MAX_LENGTH']
label_columns = config['label_columns']

def predict_toxicity(text):
    # Preprocessing
    preprocessed_text = preprocess_text(text)
    
    # Tokenizzazione e padding
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH)
    
    # Predizione
    prediction = model.predict(padded_sequence)[0]
    
    # Formattazione del risultato
    result = {}
    for i, label in enumerate(label_columns):
        result[label] = float(prediction[i])
    
    return result

# Input dall'utente e analisi
while True:
    user_input = input("Inserisci un commento da analizzare (o 'q' per uscire): ")
    if user_input.lower() == 'q':
        break
    
    toxicity_scores = predict_toxicity(user_input)
    
    print("\nRisultati dell'analisi:")
    for label, score in toxicity_scores.items():
        print(f"{label}: {score:.4f}")
    print("\n")

print("Grazie per aver usato l'analizzatore di commenti tossici!")