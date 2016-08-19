import numpy as np
import matplotlib.pyplot as plt

import json

f = open("trainrec.txt","r")

history = json.load(f)

f.close()

plt.plot(history[0])
plt.plot(history[1])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

    
model.save_weights("Classc.h5", overwrite=True)
