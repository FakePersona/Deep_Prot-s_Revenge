import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go

import json

f = open("pairs.txt", 'r')

stored = json.load(f)
f.close()

df = pd.DataFrame(np.array(stored))

labels = ["distance"]

for i in range(20):
    labels.append("x" + str(i))

labels = labels + ["global align", "local align", "BLOSUM", "structural distance"]

df = pd.DataFrame(np.array(stored))

data = [go.Heatmap( z=df.corr().values.tolist(), x = labels, y = labels)]


data = [go.Heatmap( z=df.corr().values.tolist())]

py.iplot(data, filename='PairOneConvHeat')
