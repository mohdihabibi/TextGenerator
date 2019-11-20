import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.utils import np_utils
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from keras.callbacks.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer

christmasCarol = open("christmas.txt").read()
christmasCarol = christmasCarol[1000:]
bleakHouse = open('bleakHouse.txt').read()
bleakHouse = bleakHouse[1000:]
oliver1 = open('oliver1.txt').read()
oliver1 = oliver1[1000:]
oliver2 = open('oliver2.txt').read()
oliver2 = oliver2[1000:]
oliver3 = open('oliver3.txt').read()
oliver3 = oliver3[1000:]
expectations = open('expectations.txt').read()
expectations = expectations[1000:]
copperfield = open('copperfield.txt').read()
copperfield = copperfield[1000:]
file = christmasCarol+bleakHouse+oliver1+oliver2+oliver3+expectations+copperfield


def clean_up(input):
    input = input.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    filtered = filter(lambda token: token.isalpha(), tokens)
    return " ".join(filtered)

processed_inputs = clean_up(file)

word2count = {}
total_words = 0
for word in processed_inputs.split():
    if word not in word2count:
        word2count[word] = 1
    else:
        word2count[word] += 1
    total_words += 1


word10 = []
threshold = 10
for word, count in word2count.items():
    if count >= threshold:
        if len(word) > 1:
            word10.append(word)
print(word10)         
# Removing the words from each string which appear less than 15 times
data_10 = []
for word in processed_inputs.split():
    if word in word10:
        data_10.append(word)
processed_inputs = " ".join(data_10)


tokens = processed_inputs.split(" ")
len_of_train = 3+1
fix_length_seq = []
for i in range(len_of_train,len(tokens)):
    seq = tokens[i-len_of_train:i]
    fix_length_seq.append(seq)

seq_count = {}
count_of_words = 1
for i in range(len(tokens)):
    if tokens[i] not in seq_count:
        seq_count[tokens[i]] = count_of_words
        count_of_words += 1

tokenize = Tokenizer()
tokenize.fit_on_texts(fix_length_seq)
sequences = tokenize.texts_to_sequences(fix_length_seq)   
size = len(tokenize.word_counts)

print(size)

num_of_sequence = np.empty([len(sequences),len_of_train], dtype='int32')
for i in range(len(sequences)):
    num_of_sequence[i] = sequences[i]

#split the data
X = num_of_sequence[:,:-1]
Y = num_of_sequence[:,-1]
from keras.utils import to_categorical
Y = to_categorical(Y, num_classes=size+1)
len_of_sequences = X.shape[1]

def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len,input_length=seq_len))
    model.add(LSTM(50))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(vocabulary_size,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

model = create_model(size+1,len_of_sequences)
path = './model'
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit(X,Y,batch_size=128,epochs=500,verbose=1,callbacks=[checkpoint])
dump(tokenizer,open('tokenizer','wb')) 