from neural_nets_from_scratch import *
import neural_nets_from_scratch
from matplotlib import pyplot as plt
import numpy as np

x = np.arange(-1000, 1000, 1)
y = []
for X in x:
    y.append(X**2)
x = x.reshape((2000, 1))
y = np.array(y).reshape((2000, 1))
model = Network(layers=[Layer(1, 50, Sigmoid()), Layer(50, 50, Sigmoid()), Layer(50, 50, Sigmoid()), Layer(50, 50), Layer(50, 1)])
model.train(x, y, epochs=100, batch_size=0)
plt.plot(x, y)
plt.plot(x, model.predict(x))
plt.show()