from __future__ import print_function

import numpy as np

from keras import backend as K

from keras.models import Sequential
from keras.layers import recurrent, RepeatVector, Activation, TimeDistributed, Dense, Dropout

from Bio import SeqIO

def dist2(u,v):
    return sum([(x-y)**2 for (x, y) in zip(u, v)])

class AcidEmbedding(object):

    def __init__(self, maxlen):
        self.maxlen = maxlen

        self.chars = list('rndeqkstchmavgilfpwybzuxXo')

        self.embed = [[-1, 1, 9.09, 71.8],
                      [-1, 0, 8.8, 2.4],
                      [-1, -1, 9.6, 0.42],
                      [-1, -1, 9.67, 0.72],
                      [-1, 0, 9.13, 2.6],
                      [-1, 1, 10.28, 200],
                      [0, 0, 9.15, 36.2],
                      [0, 0, 9.12, 200],
                      [1, 0, 10.78, 200],
                      [-1, 1, 8.97, 4.19],
                      [1, 0, 9.21, 5.14],
                      [1, 0, 9.87, 5.14],
                      [1, 0, 9.72, 5.6],
                      [0, 0, 9.6, 22.5],
                      [1, 0, 9.76, 3.36],
                      [1, 0, 9.6, 2.37],
                      [1, 0, 9.24, 2.7],
                      [-1, 0, 10.6, 1.54],
                      [0, 0, 9.39, 1.06],
                      [-1, 0, 9.11, 0.038],
                      [1, 0.5, 8.95, 37.1],
                      [1, -0.5, 9.40, 1.66],
                      [250, -250, 250, -250],
                      [0, 0, 0, 0],
                      [500, 500, 500, 500],
                      [-500, -500, -500, -500]]
        
        self.embed = [[x/500 for x in X] for X in self.embed]

        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, 4))
        for i, c in enumerate(C):
            X[i] = self.embed[self.char_indices[c]]
        return X


    def decode(self, X):
        prob = [[-dist2(x, y) for y in self.embed] for x in X]
        prob = (np.array(prob)).argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in prob)

chars = 'rndeqkstchmavgilfpwybzuxXo'
ctable = AcidEmbedding(11)

ACIDS = 4
encoding_dim = 50

np.set_printoptions(threshold=np.nan)

print("Generating data...")

data = []
test = []

record = SeqIO.parse("../../../data/smallData.fa", "fasta")

ind = 0
for rec in record:
    ind +=1
    if len(test) > 229999:
        break
    if ind > 25502:
        break
    if ((len(data) + len(test)) % 6) == 5:
        for k in range(len(rec.seq)/3 - 10):
            test.append([rec.seq[3 * k + i] for i in range(11)])
    else:
        for k in range(len(rec.seq)/3 - 4):
            data.append([rec.seq[3 * k + i] for i in range(11)] )

X = np.zeros((len(data), 11, 4))

for i, sentence in enumerate(data):
    X[i] = ctable.encode(sentence, maxlen=11)

X_val = np.zeros((len(test), 11, 4))

for i, sentence in enumerate(test):
    X_val[i] = ctable.encode(sentence, maxlen=11)

print("Creating model...")
model = Sequential()

#Recurrent encoder
model.add(recurrent.LSTM(encoding_dim, init='he_uniform', input_shape=(11, ACIDS), return_sequences=True, dropout_W=0.1, dropout_U=0.1))
model.add(recurrent.LSTM(encoding_dim, init='he_uniform', return_sequences=True, dropout_W=0.1, dropout_U=0.1))
model.add(recurrent.LSTM(encoding_dim, init='he_uniform', dropout_W=0.1, dropout_U=0.1))

model.add(RepeatVector(11))

#And decoding
model.add(recurrent.LSTM(ACIDS, return_sequences=True))

model.add(TimeDistributed(Dense(ACIDS)))

#model.load_weights("RecWind.h5")

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
        preds = model.predict(row, verbose=0)
        correct = ctable.decode(row[0])
        guess = ctable.decode(preds[0])
        print('T', correct)
        print('P', guess)
        print('---')

model.save_weights("RecWind.h5", overwrite=True)
