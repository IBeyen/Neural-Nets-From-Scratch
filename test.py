from networks_and_layers import *
from matplotlib import pyplot as plt
import numpy as np


x = np.arange(-1000,1000)
y = x**2 + x/2 + 6
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
model = Network(layers=[Layer(1, 50, Relu()), Layer(50, 50, Relu()), Layer(50, 50, Relu()), Layer(50, 1, Relu())], optimizer=RMSProp())
w1 = np.copy(model.layers[3].weights)
model.train(x, y, epochs=200, learning_rate=5e-4, batch_size=64)
plt.plot(x, y)
plt.plot(x, model.predict(x))
plt.show()
plt.plot(np.arange(1,201), model.loss)
plt.show()
w2 = model.layers[3].weights
print(w1[0][0]-w2[0][0])
print(w1[0][0])
print(w2[0][0])