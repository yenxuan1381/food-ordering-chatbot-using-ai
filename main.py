import random
import json
import pickle

import numpy as np
import telebot
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('training\intents.json').read())
menu = json.loads(open('training\menu.json').read())

words = pickle.load(open('training\words.pkl', 'rb'))
labels = pickle.load(open('training\labels.pkl', 'rb'))
model = load_model('training\chatbot_model.h5')


isOrder = False
total_price = 0.00

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

    list_of_intents = intents_json['intents']
    for i in list_of_intents:

        if i['tag'] == tag:
            result = random.choice(i['responses'])

            break
    return result


def order_food(menu_json, id):
    for x in menu_json:

        if x["food_id"] == id:
            print("Food ID:", "".join(x["food_id"]))
            print("Item name:", "".join(x["item_name"]))
            print("Price:", "".join(x["price"]))
            return float(x["price"])


def print_stall_menu(menu_json, stall, delivery):
    for x in menu_json:

        # prints menu for delivery and for the stall
        if delivery:
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


def add_order(menu_json, order_id, cart):
    input_dict = json.loads(menu_json)
    output_dict = [x for x in input_dict if x['food_id'] == order_id]
    res = json.dumps(output_dict)

    cart.append(res)

    return cart


# API_KEY = os.getenv('API_KEY')
bot = telebot.TeleBot('2129062414:AAE4c6pa44_p0t2BQTDDRuDkr6xd_do8GWo')


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Hi! How can I help you? Would recommend you to start viewing the menu first.")

# @bot.message_handler(commands=['order'])
# def start(message):
#     bot.send_message(message.chat.id, "Hi! How can I help you? Would recommend you to start viewing the menu first.")
#
#
#
#     while temp:
#         bot.send_message(message.chat.id, 'Type the food id of the food that u want:')
#         food_id = input()
#         price = order_food(menu, food_id)
#         total_price += price  # Total price calculation here
#
#         print('Would you like to order anymore food? (1 = no)')
#         flag = input()
#
#         if flag == '1':
#             temp = False
#
#     print('Thanks for your order!')
#     print('The total price is ')
#     print(total_price)


# shopping_cart_price = 0.00


@bot.message_handler(func=lambda m: True)
def ordering_process(message):
    delivery_service = False
    global isOrder
    global total_price

    msg = message
    msg2 = message.text

    if isOrder:
        food_id = message.text
        bot.send_message(msg.chat.id, food_id)
        if food_id.isnumeric():

            price = order_food(menu, food_id)

            total_price += price  # Total price calculation here

            bot.send_message(msg.chat.id, "Total Price: " +str(total_price))

        else:
            bot.send_message(msg.chat.id, "You did not enter an integer")

        isOrder=False
    else:
        ints = predict_class(msg2)
        res = get_response(ints, intents)
        # print(res)
        bot.send_message(msg.chat.id, res)

        if res == "Sure, we have menu for delivery only":
            delivery_service = True
        isOrder = False

        # print out the mamak menu when the user ask for it
        if res == "Ok. I will fetch a Mamak menu for u":

            if delivery_service:
                mamak_delivery_menu = open('menu_folder/mamakmenudelivery.pdf', 'rb')
                bot.send_document(msg.chat.id, mamak_delivery_menu)
            else:
                mamak_menu = open('menu_folder/mamakmenu.pdf', 'rb')
                bot.send_document(msg.chat.id, mamak_menu)

            isOrder = False

        # print Japanese menu
        if res == "Ok. I will fetch a Japanese menu for u":
            if delivery_service:
                japanese_delivery_menu = open('menu_folder/japanesemenudelivery.pdf', 'rb')
                bot.send_document(msg.chat.id, japanese_delivery_menu)
            else:
                japanese_menu = open('menu_folder/japanesemenu.pdf', 'rb')
                bot.send_document(msg.chat.id, japanese_menu)

            isOrder = False

        # print Korean menu
        if res == "Ok. I will fetch a Korean menu for u":
            if delivery_service:
                korean_delivery_menu = open('menu_folder/koreanmenudelivery.pdf', 'rb')
                bot.send_document(msg.chat.id, korean_delivery_menu)
            else:
                korean_menu = open('menu_folder/koreanmenu.pdf', 'rb')
                bot.send_document(msg.chat.id, korean_menu)

            isOrder = False

        # print beverage menu
        if res == "Ok. I will fetch a beverage menu for u":
            if delivery_service:
                beverage_delivery_menu = open('menu_folder/beveragemenudelivery.pdf', 'rb')
                bot.send_document(msg.chat.id, beverage_delivery_menu)
            else:
                beverage_menu = open('menu_folder/beveragemenu.pdf', 'rb')
                bot.send_document(msg.chat.id, beverage_menu)

            isOrder = False

        # print Malay menu
        if res == "Ok. I will fetch a Malay menu for u":
            if delivery_service:
                malay_delivery_menu = open('menu_folder/malaymenudelivery.pdf', 'rb')
                bot.send_document(msg.chat.id, malay_delivery_menu)
            else:
                malay_menu = open('menu_folder/malaymenu.pdf', 'rb')
                bot.send_document(msg.chat.id, malay_menu)

            isOrder = False

        if res == "Ok. Please type the food id of the food you would like to order.":
            isOrder = True




    # if res == "Ok. What food would you like to order?":
    #     total_price = 0.00
    #     temp = True
    #     while temp:
    #         print('Type the food id of the food that u want:')
    #         food_id = input()
    #         price = order_food(menu, food_id)
    #         total_price += price  # Total price calculation here
    #
    #         print('Would you like to order anymore food? (1 = no)')
    #         flag = input()
    #
    #         if flag == '1':
    #             temp = False
    #
    #     print('Thanks for your order!')
    #     print('The total price is ')
    #     print(total_price)

        if res == "See you and come back if you want to chat with me again." or res == "Talk to you later" or res == "Goodbye!":
            total_price = 0
            exit(0)


bot.infinity_polling()
