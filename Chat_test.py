import pickle

import matplotlib.pyplot as plt
from keras.models import load_model
model = load_model('model/chatbot_model.h5')
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
from PIL import Image

intents = json.loads(open('model/intents.json',encoding='utf-8').read())
words = pickle.load(open('model/words1.pkl','rb'))
classes = pickle.load(open('model/classes1.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        #if(i['tag']== tag) and tag == 'Informazioni':
        #    result = random.choice(i['responses'])
        #    image = Image.open("c:/Users/Walter/Desktop/Carditello/scalone.jpg")
            #image.show()
        #    return result,image
        #    break
        #elif (i['tag']== tag) and tag != 'Informazioni':
        #  result = random.choice(i['responses'])
        #  break
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

print("TEST")

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    print(res)
    return res

#while True:
#    text = input("")
#    chatbot_response(text)

#lista = ints[0]
#    lista_test = lista["intent"]
#    if lista_test == 'Informazioni':
#       immagine = Image.open('c:/Users/Walter/Desktop/Carditello/scalone.jpg')
#       immagine = immagine.show()
#    else:
#       immagine = None
#       return immagine

#while True:
#    text = input("")
#    ints = predict_class(text, model)
#    print(type(ints))#list
#    print(type(ints[0]))#dizionario
#    lista = ints[0]
#    print(lista)
#    lista_test = lista["intent"]
#    if lista_test == 'Informazioni':
#        immagine = Image.open('c:/Users/Walter/Desktop/Carditello/scalone.jpg')
#        immagine.show()
#    else:
#        immagine = None
#    res = getResponse(ints,intents)
#    print(res)
