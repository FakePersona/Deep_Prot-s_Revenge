import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go

import json

f = open("Single.txt", 'r')

stored = json.load(f)
f.close()

df = pd.DataFrame(np.array(stored))

data = [go.Heatmap( z=df.corr().values.tolist())]

py.plot(data, filename='SingleOneConvHeat')
