import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go

import json

f = open("pairs.txt", 'r')

stored = json.load(f)
f.close()

labels = ["Norme"]

for i in range(50):
    labels.append("x" + str(i))

labels = labels + ["global align", "", "local align", "BLOSUM align", "Structural distance"]

df = pd.DataFrame(np.array(stored))

data = [go.Heatmap( z=df.corr().values.tolist(), x = labels, y = labels)]

py.plot(data, filename='PairEmbedRecHeat')
