import random
import json
import pickle
import numpy as np


import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('popular')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json", encoding="utf8").read())


words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')

def _clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def _bag_of_words(sentence, words, show_details=True):
    sentence_words = _clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def _predict_class(sentence, model):
    p = _bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def _get_response(ints, intents_json):
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag']  == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = _predict_class(msg, model)
    res = _get_response(ints, intents)
    return res


from flask import Flask, render_template, request

chatbot = Flask(__name__)
chatbot.static_folder = 'static'

@chatbot.route("/")
def home():
    return render_template("index.html")

@chatbot.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    chatbot.run()
