import random
import json
import pickle
import numpy as np
import ast
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
menu = json.loads(open('menu.json').read())

words = pickle.load(open('words.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))
model = load_model('chatbot_model.h5')


# print(labels)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bagofwords(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bagofwords(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': labels[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    # print(tag)
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        # print(i['tag'])
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            # print(result)
            break
    return result


# def get_stall(menu_json):
#     # TODO: retrive the stall


# def get_food(menu_json, stall_name):
#     # TODO: return the food for a specific stall


# def delivery_services(menu_json):
#     # TODO: return the menu with only delivery services
#
# def error_detection():
#     # TODO: error detection for inputs

print("Hi! How can I help you")


# while True:

# message = input("")
# # print(message)
# ints = predict_class(message)
# res = get_response(ints, intents)
# print(res)
