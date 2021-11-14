import random
import json
import pickle
import numpy as np
import ast
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


# predict the class based on the sentence
def predict_class(sentence):
    bow = bagofwords(sentence)
    res = model.predict(np.array([bow]))[0]

    # allows some uncertainty (error detection)
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # sort the results
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


# def total_price(menu_json, stall_name):
#     # TODO: return the total price

def print_stall_menu(menu_json, stall, delivery):
    # TODO: retrive the stall menu

    for x in menu_json:

        # prints menu for delivery and for the stall
        if delivery == True:
            if x["stall_name"] == stall and x["delivery_service"] == 'yes':
                print("Food ID:", "".join(x["food_id"]))
                print("Stall name:", "".join(x["stall_name"]))
                print("Item name:", "".join(x["item_name"]))
                print("Price:", "".join(x["price"]))
                print("Delivery Service:", "".join(x["delivery_service"]))
                print("\n")

        # prints menu for stall only
        else:
            if x["stall_name"] == stall:
                print("Food ID:", "".join(x["food_id"]))
                print("Stall name:", "".join(x["stall_name"]))
                print("Item name:", "".join(x["item_name"]))
                print("Price:", "".join(x["price"]))
                print("Delivery Service:", "".join(x["delivery_service"]))
                print("\n")


def add_order(menu_json, order_id, shopping_cart):
    input_dict = json.loads(menu_json)
    output_dict = [x for x in input_dict if x['food_id'] == order_id]
    res = json.dumps(output_dict)

    shopping_cart.append(res)

    return shopping_cart


print("Hi! How can I help you")

while True:
    message = input("")
    delivery_service = False
    shopping_cart = []

    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)

    if res == "Sure, we have menu for delivery only":
        delivery_service = True

    # print out the mamak menu when the user ask for it
    if res == "Ok. I will fetch a Mamak menu for u":
        print_stall_menu(menu, 'Mamak', delivery_service)

        # print_stall_menu(menu, 'Japanese')
        # print_stall_menu(menu, 'Korean')
        # print_stall_menu(menu, 'Beverage')
        # print_stall_menu(menu, 'Malay')
