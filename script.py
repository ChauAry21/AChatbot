import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
import matplotlib.pyplot as plt

nltk.download(['punkt', 'punkt_tab', 'wordnet', 'stopwords', 'omw-1.4'])
dataset_path = './intents.json'
print("Path to dataset files:", dataset_path)

with open(dataset_path, 'r') as file:
    data = json.load(file)

words, classes, documents = [], [], []

for intent in data['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

lem = WordNetLemmatizer()
ignore = ['.', '?', '!', ',']
stopwrd = set(stopwords.words('english'))

words = [lem.lemmatize(w.lower()) for w in words if w not in stopwrd and w.lower() not in ignore]
words = sorted(set(words))
classes = sorted(set(classes))

print("Sample documents:", documents[:2])
print("Classes:", classes)

with open('words.pkl', 'wb') as file:
    pickle.dump(words, file)

with open('classes.pkl', 'wb') as file:
    pickle.dump(classes, file)

training = []
output_empty = [0] * len(classes)

for d in documents:
    bag = []
    pattern_words = [lem.lemmatize(w.lower()) for w in d[0]]
    bag = [1 if w in pattern_words else 0 for w in words]

    output_row = list(output_empty)
    output_row[classes.index(d[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = numpy.array(training, dtype='object')

X_train = list(training[:, 0])
y_train = list(training[:, 1])

X_train = numpy.array(X_train, dtype='float32')
y_train = numpy.array(y_train, dtype='float32')

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(32)

model = Sequential([
    Dense(256, input_shape=(len(X_train[0]),), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(y_train[0]), activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    epochs=200, verbose=1,
    validation_data=(X_val, y_val)
)

model.save('./trained_model.keras')

plt.rcParams["figure.figsize"] = (12, 8)
epochs = numpy.arange(0, 200)
plt.style.use("ggplot")
plt.figure()
plt.plot(epochs, history.history["loss"], label="train_loss")
plt.plot(epochs, history.history["val_loss"], label="val_loss")
plt.plot(epochs, history.history['accuracy'], label="accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()