from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

#import plotly.plotly as py

from keras import backend as K

import glob
import os
from os.path import basename
from Bio.PDB import *
from Bio.PDB.Polypeptide import three_to_one
from Bio import SeqIO

import json
from sklearn import cluster
from keras.models import Sequential
from keras.layers import recurrent, RepeatVector, Activation, TimeDistributed, Dense, Dropout

# Hash table

def dist2(u,v):
    return sum([(x-y)**2 for (x, y) in zip(u, v)])

class AcidEmbedding(object):

    def __init__(self, maxlen):
        self.maxlen = maxlen

        self.chars = list('rndeqkstchmavgilfpwybzuxXo')

        self.embed = [[-4.5, 1, 9.09, 71.8],
                      [-3.5, 0, 8.8, 2.4],
                      [-3.5, -1, 9.6, 0.42],
                      [-3.5, -1, 9.67, 0.72],
                      [-3.5, 0, 9.13, 2.6],
                      [-3.9, 1, 10.28, 200],
                      [-0.8, 0, 9.15, 36.2],
                      [-0.7, 0, 9.12, 200],
                      [2.5, 0, 10.78, 200],
                      [-3.2, 1, 8.97, 4.19],
                      [1.9, 0, 9.21, 5.14],
                      [1.8, 0, 9.87, 5.14],
                      [4.2, 0, 9.72, 5.6],
                      [-0.4, 0, 9.6, 22.5],
                      [4.5, 0, 9.76, 3.36],
                      [3.8, 0, 9.6, 2.37],
                      [2.8, 0, 9.24, 2.7],
                      [-1.6, 0, 10.6, 1.54],
                      [-0.9, 0, 9.39, 1.06],
                      [-1.3, 0, 9.11, 0.038],
                      [1, 0.5, 8.95, 37.1],
                      [1, -0.5, 9.40, 1.66],
                      [250, -250, 250, -250],
                      [0, 0, 0, 0],
                      [500, 500, 500, 500],
                      [-500, -500, -500, -500]]

        self.embed = [[x for x in X] for X in self.embed]

        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, 4))
        for i, c in enumerate(C):
            X[i] = self.embed[self.char_indices[c]]
        return X

    def get_charge(self, C):
        charge = 0
        for c in C:
            charge += self.embed[self.char_indices[c]][1]
        return charge

    def get_hydro(self, C):
        hydro = 0
        for c in C:
            hydro += self.embed[self.char_indices[c]][0]
        return hydro

    def decode(self, X):
        prob = [[-dist2(x, y) for y in self.embed] for x in X]
        prob = (np.array(prob)).argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in prob)


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

# Initial parameters
    
chars = 'rndeqkstchmavgilfpwybzuxXo'
ctable = CharacterTable(chars, 11)
lookup = AcidEmbedding(11)

ACIDS = 26
encoding_dim = 20

# Data Generating

print("Generating data...")

data = []
test = []

dataNames = []



# Parsing fasta file

record = SeqIO.parse("../../../data/smallData.fa", "fasta")

proteins = []
UniqNames = []

for rec in record:
    proteins.append(rec.seq)
    UniqNames.append(rec.name)

nameInd = dict((c, i) for i, c in enumerate(UniqNames))

print(len(UniqNames))

record = SeqIO.parse("../../../data/smallData.fa", "fasta")

# parsing PDB files

print("parsing PDB")

PDB_list = glob.glob("../../../../PDBMining/*/*.ent")

p = PDBParser()
secondaryStruct = []
Valid = [False for _ in proteins]
PDBNames = []
for f in PDB_list:
    name = os.path.splitext(basename(f))[0]
    PDBNames.append(name)
    struct = p.get_structure(name,f)
    res_list = Selection.unfold_entities(struct, 'R')
    try:
        seq = [three_to_one(a.get_resname()).lower() for a in res_list]
    except (KeyError):
        seq = []
    try:
        if seq == [a for a in proteins[nameInd[name]]]:
            Valid[nameInd[name]] = True
    except KeyError:
        pass
    struct_dssp = p.get_structure(name,f)
    try:
        dssp = DSSP(struct_dssp[0], f)
    except Exception:
        Valid[nameInd[name]] = False
    a_keys = list(dssp.keys())
    sec = [dssp[a][2] for a in a_keys]
    try:
        if len(sec) != len(proteins[nameInd[name]]):
            Valid[nameInd[name]] = False
    except KeyError:
        pass
    secondaryStruct.append(sec)

print(len(secondaryStruct))
print(len(PDBNames))

PDBInd = dict((c, i) for i, c in enumerate(PDBNames))

dataSecond = []

print("parsing sequences")

ind = -1
for rec in record:
    ind +=1
    if ind > 13000:
        break
    if not Valid[nameInd[rec.name]]:
        continue
    for k in range(len(rec.seq) // 3 - 4):
        data.append([rec.seq[3 * k + i] for i in range(11)])
        dataSecond.append([secondaryStruct[PDBInd[rec.name]][3 * k + i] for i in range(11)])
        dataNames.append(rec.name)


# Encoding data

X = np.zeros((len(data), 11, len(chars)), dtype=np.bool)

for i, sentence in enumerate(data):
    X[i] = ctable.encode(sentence, maxlen=11)


print("Creating model...")
model = Sequential()

#Recurrent encoder
model.add(recurrent.LSTM(encoding_dim, input_shape=(11, ACIDS), return_sequences=True, dropout_W=0.1, dropout_U=0.1))
model.add(recurrent.LSTM(encoding_dim, return_sequences=True, dropout_W=0.1, dropout_U=0.1))
model.add(recurrent.LSTM(encoding_dim, dropout_W=0.1, dropout_U=0.1))

model.add(RepeatVector(11))

#And decoding
model.add(recurrent.LSTM(ACIDS, return_sequences=True))

model.add(TimeDistributed(Dense(ACIDS)))

model.add(Activation('softmax'))

model.load_weights("RecOne.h5")

model.compile(optimizer='rmsprop', loss='binary_crossentropy')

get_summary = K.function([model.layers[0].input, K.learning_phase()], [model.layers[2].output])

print("Let's go!")

Embed = [[0 for _ in range(encoding_dim)] for _ in range(len(X))]

for i in range(len(X)):
    row = X[np.array([i])]
    preds = model.predict_classes(row, verbose=0)
    correct = ctable.decode(row[0])
    intermediate = get_summary([row, 0])[0][0]
    Embed[i] = intermediate

# Preparing data for correlating

Properties = np.zeros((len(data), 30))

for i in range(len(Properties)):
    #Norm
    Properties[i][0] = np.linalg.norm(Embed[i])
    #Dimensional orientation
    for k in range(encoding_dim):
        Properties[i][k+1] = Embed[i][k]
    # Hydropathy and charge
    Properties[i][21] = lookup.get_hydro(data[i])
    Properties[i][22] = lookup.get_charge(data[i])
    # Aliphatic
    for c in data[i]:
        if c in 'ailv':
            Properties[i][23] += 1
    # Aromatic
    for c in data[i]:
        if c in 'fwy':
            Properties[i][24] += 1
    # Neutral
    for c in data[i]:
        if c in 'ncqmst':
            Properties[i][25] += 1
    # Acidic
    for c in data[i]:
        if c in 'de':
            Properties[i][26] += 1
    # Acidic
    for c in data[i]:
        if c in 'rhk':
            Properties[i][27] += 1
    # Secondary Structure
    s = 0
    for c in dataSecond[i]:
        if c == 'H':
            s += 1
        if c == 'B':
            s -= 1
    if s == 11:
        Properties[i][28] = 1
    Properties[i][29] = s
        
f = open("Single.txt", 'w')
json.dump(Properties.tolist(), f)
f.close()
