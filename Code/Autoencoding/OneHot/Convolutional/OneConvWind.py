from __future__ import print_function

import numpy as np

from keras import backend as K

from keras.models import Sequential
from keras.layers import recurrent, RepeatVector, Activation, TimeDistributed, Dense, Dropout, Convolution1D, Flatten

from Bio import SeqIO

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

chars = 'rndeqkstchmavgilfpwybzuxXo'
ctable = CharacterTable(chars, 11)

ACIDS = 26
encoding_dim = 20

np.set_printoptions(threshold=np.nan)

print("Generating data...")

data = []
test = []

record = SeqIO.parse("../../../data/smallData.fa", "fasta")

ind = 0
for rec in record:
    ind +=1
    if len(test) > 499999:
        break
    if ind > 25502:
        break
    if ((len(data) + len(test)) % 6) == 5:
        for k in range(len(rec.seq)/3 - 10):
            test.append([rec.seq[3 * k + i] for i in range(11)])
    else:
        for k in range(len(rec.seq)/3 - 4):
            data.append([rec.seq[3 * k + i] for i in range(11)] )

print(ind)
            
X = np.zeros((len(data), 11, 26))

for i, sentence in enumerate(data):
    X[i] = ctable.encode(sentence, maxlen=11)

X_val = np.zeros((len(test), 11, 26))

for i, sentence in enumerate(test):
    X_val[i] = ctable.encode(sentence, maxlen=11)

print("Creating model...")
model = Sequential()

#Recurrent encoder
model.add(Convolution1D(10, 3, activation='relu', input_shape=(11, ACIDS)))
model.add(Dropout(0.2))

model.add(Convolution1D(10, 2, activation='relu'))
model.add(Dropout(0.1))

model.add(Convolution1D(13, 2, activation='relu'))
model.add(Dropout(0.1))

model.add(Convolution1D(3, 2, activation='relu'))
model.add(Dropout(0.1))

model.add(Convolution1D(6, 2, activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(encoding_dim))

model.add(RepeatVector(11))

#And decoding
model.add(recurrent.LSTM(ACIDS, return_sequences=True))

model.load_weights("ConvOne.h5")

model.compile(optimizer='rmsprop', loss='mse')

print("Let's go!")
# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 30):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, X, batch_size=128, nb_epoch=1,
              validation_data=(X_val, X_val))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        row = X_val[np.array([ind])]
        preds = model.predict_classes(row, verbose=0)
        correct = ctable.decode(row[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('T', correct)
        print('P', guess)
        print('---')

model.save_weights("ConvOne.h5", overwrite=True)
