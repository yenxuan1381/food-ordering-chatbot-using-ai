import json
import random
import pickle
import numpy as np

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
labels = []
documents = []
ignore = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)

        words.extend(word_list)

        documents.append((word_list, intent['tag']))
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]  # edited
words = sorted(list(set(words)))

labels = sorted(list(set(labels)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))

train_set = []
output_empty = [0] * len(labels)

for doc in documents:
    bag = []
    word_patterns = doc[0]

    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[labels.index(doc[1])] = 1
    train_set.append([bag, output_row])

random.shuffle(train_set)
train_set = np.array(train_set)

train_x = list(train_set[:, 0])
train_y = list(train_set[:, 1])

model = Sequential(
    [
        Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(train_y[0]), activation='softmax')
    ]
)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("Done")
