from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K

from keras.models import Sequential
from keras.layers import recurrent, RepeatVector, Activation, TimeDistributed, Dense, Dropout, Convolution1D, Flatten

from Bio import SeqIO

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
ctable = AcidEmbedding(150)

ACIDS = 26
encoding_dim = 50

np.set_printoptions(threshold=np.nan)

print("Generating data...")

data = []
test = []

record = SeqIO.parse("../data/smallData.fa", "fasta")

Labels = [[1, 0, 0, 0] for _ in range(1, 2440)] + [[0, 1, 0, 0] for _ in range(2440, 5236)] + [[0, 0, 1, 0] for _ in range(5236, 9205)] + [[0, 0, 0, 1] for _ in range(9205, 12551)]

ind = -1

lab_data = []
lab_test = []
for rec in record:
    ind +=1
    if len(rec.seq) > 150:
        continue
    if ind > 12500:
        break
    if ((len(data) + len(test)) % 7) == 5:
        test.append([a for a in rec.seq] + ['o' for _ in range(150 - len(rec.seq))])
        lab_test.append(Labels[ind])
    else:
        data.append([a for a in rec.seq] + ['o' for _ in range(150 - len(rec.seq))])
        lab_data.append(Labels[ind])

print(ind)
            
X = np.zeros((len(data), 150, 4))

for i, sentence in enumerate(data):
    X[i] = ctable.encode(sentence, maxlen=150)

X_val = np.zeros((len(test), 150, 4))

for i, sentence in enumerate(test):
    X_val[i] = ctable.encode(sentence, maxlen=150)

F = []

for x in X:
    F.append([[x[k + i] for i in range(11)] for k in range(len(x) - 10)])

F_val = []

for x in X_val:
    F_val.append([[x[k + i] for i in range(11)] for k in range(len(x) - 10)])

print(len(F))
print(len(F[0]))
print("Creating dummy model...")
dum = Sequential()

#Recurrent encoder
dum1 = recurrent.LSTM(encoding_dim, input_shape=(11, 4), return_sequences=True, dropout_W=0.1, dropout_U=0.1)
dum.add(dum1)

dum2 = recurrent.LSTM(encoding_dim, return_sequences=True, dropout_W=0.1, dropout_U=0.1)
dum.add(dum2)

dum3 = recurrent.LSTM(encoding_dim, dropout_W=0.1, dropout_U=0.1)
dum.add(dum3)

dum.add(RepeatVector(11))

#And decoding
dum.add(recurrent.LSTM(4, return_sequences=True))

dum.add(TimeDistributed(Dense(4)))

dum.load_weights("RecWind.h5")

print("Creating actual model")

model = Sequential()

model.add(TimeDistributed(dum1, input_shape = (140, 11, 4)))
model.add(TimeDistributed(dum2))
model.add(TimeDistributed(dum3))

model.add(Convolution1D(30, 5, activation='relu'))
model.add(Dropout(0.2))

model.add(Convolution1D(10, 2, activation='relu'))
model.add(Dropout(0.1))

model.add(Convolution1D(3, 2, activation='relu'))
model.add(Dropout(0.1))

model.add(Convolution1D(6, 2, activation='relu'))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(4))

model.add(Activation('softmax'))

#model.load_weights("Class.h5")

model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])

print("Let's go!")
history = model.fit(F, lab_data, batch_size=128, nb_epoch=100, validation_data=(F_val, lab_test))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    # for i in range(10):
    #     ind = np.random.randint(0, len(F_val))
    #     row = F_val[np.array([ind])]
    #     preds = model.predict_classes(row, verbose=0)
    #     correct = np.array(lab_test[ind]).argmax()
    #     guess = preds[0]
    #     print('T', correct)
    #     print('P', guess)
    #     print('---')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

    
model.save_weights("Classc.h5", overwrite=True)
