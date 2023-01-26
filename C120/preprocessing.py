import nltk
from nltk.stem import PorterStemmer
import numpy as np
import json
import pickle

train_data=open('intents.json').read()


intents=json.loads(train_data)['intents']


stemmer=PorterStemmer()

words=[]
words_tags_list=[]
classes=[]

for intent in intents:

    for pattern in intent['patterns']:
        pattern_word=nltk.word_tokenize(pattern)
        words.extend(pattern_word)
        words_tags_list.append((pattern_word,intent['tag']))

    if intent['tag'] not in classes:
        classes.append(intent['tag'])

ignore_words=['?','!',',','.',"'s","'m"]
stem_words=[]

def get_stem_words(words,ignore_words):

    for word in words:
        if word not in ignore_words:
            w=stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words

stem_words=get_stem_words(words,ignore_words)

stem_words=sorted(list(set(stem_words)))
classes=sorted(list(set(classes)))

pickle.dump(stem_words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training_data=[]
number_of_tags=len(classes)
labels=[0]*number_of_tags
print(classes)
for word_tag in words_tags_list:

    bag_of_words=[]
    pattern_words=word_tag[0]
    temp=[]

    for word in pattern_words:
        if word not in ignore_words:
            w=stemmer.stem(word.lower())
            temp.append(w)


    
    for word in stem_words:
        if word in temp:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    
    bag_of_words-np.array(bag_of_words)
    labels_encoding=list(labels)
    tag=word_tag[1]
    tag_index=classes.index(tag)
    labels_encoding[tag_index]=1   

    training_data.append([bag_of_words,labels_encoding])


training_data=np.array(training_data,dtype=object)

train_x=list(training_data[:,0])
train_y=list(training_data[:,1])