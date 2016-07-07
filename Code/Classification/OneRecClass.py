from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

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
ctable = CharacterTable(chars, 150)

ACIDS = 26
encoding_dim = 20

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
            
X = np.zeros((len(data), 150, 26))

for i, sentence in enumerate(data):
    X[i] = ctable.encode(sentence, maxlen=150)

X_val = np.zeros((len(test), 150, 26))

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
dum1 = recurrent.LSTM(encoding_dim, input_shape=(11, ACIDS), return_sequences=True, dropout_W=0.1, dropout_U=0.1)
dum.add(dum1)

dum2 = recurrent.LSTM(encoding_dim, return_sequences=True, dropout_W=0.1, dropout_U=0.1)
dum.add(dum2)

dum3 = recurrent.LSTM(encoding_dim, dropout_W=0.1, dropout_U=0.1)
dum.add(dum3)

dum.add(RepeatVector(11))

#And decoding
dum.add(recurrent.LSTM(ACIDS, return_sequences=True))

dum.add(TimeDistributed(Dense(ACIDS)))

dum.add(Activation('softmax'))

dum.load_weights("RecOne.h5")

print("Creating actual model")

model = Sequential()

model.add(TimeDistributed(dum1, input_shape = (140, 11, ACIDS)))
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

    
model.save_weights("Class.h5", overwrite=True)
