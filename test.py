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
model = Network(layers=[Layer(1, 50), Layer(50, 100, Sigmoid()), Layer(100, 50), Layer(50, 25), Layer(25, 1)])
model.train(x, y, epochs=155, learning_rate=1e-10, batch_size=32)
plt.plot(x, y)
plt.plot(x, model.predict(x))
plt.show()
plt.plot(np.arange(1, 156), model.loss)
plt.show()