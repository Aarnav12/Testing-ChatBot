import nltk
import numpy as np
import json
import pickle
import tensorflow
import random
from preprocessing import *

model = tensorflow.keras.models.load_model('C:/Users/sapsl/Downloads/C120/chatbot_model.h5')

ignore_words=['?','!',',','.',"'s","'m"]

intents = json.loads(open('C:/Users/sapsl/Downloads/C120/intents.json').read())['intents']

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl','rb'))

def preprocess_user_input(user_input):
    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words)
    input_word_token_2=sorted(list(set(input_word_token_2)))

    bag = []
    bag_of_words = []

    for word in words:
        if word in input_word_token_2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)

    return np.array(bag)

def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])

    return predicted_class_label

def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)
    predicted_class = classes[predicted_class_label]

    for intent in intents:

        if intent['tag'] == predicted_class:
            bot_response = random.choice(intent['responses'])

            return bot_response

print("Hi I'm Chulbul")
print("How can I help you ? ")

while True:
    user_input = input("Please express your feeling : ")
    print("User Input : ", user_input)
    response = bot_response(user_input)
    print("Bot Response : ", response)