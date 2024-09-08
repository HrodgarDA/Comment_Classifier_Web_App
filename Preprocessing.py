import os
import pickle
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Comments_Preprocessing import preprocess_text

nltk.download('stopwords')

#model path
model_dir_path = '/Users/rugg/Documents/GitHub/Comment_Classifier_Web_App/toxic_comment_model.h5'

#Loading
model = tf.keras.models.load_model(os.path.join(model_dir_path, 'RNN_model.h5')) #Model

with open(os.path.join(model_dir_path, 'tokenizer.pickle'), 'rb') as handle: #Tokenizer
    tokenizer = pickle.load(handle)

with open(os.path.join(model_dir_path, 'model_parameters.json'), 'r') as f: #Parameters
    config = json.load(f)

MAX_LENGTH = config['MAX_LENGTH']
label_columns = config['label_columns']

#Custom functions

def preprocess_text(text):  #lower case conversion, Removal of special characters and stopwords
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

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