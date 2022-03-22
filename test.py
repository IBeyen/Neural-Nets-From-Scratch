from networks_and_layers import *
from matplotlib import pyplot as plt
import numpy as np


x = np.arange(-1000,1000)
y = x**2 + x/2 + 6
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
model = Network(layers=[Layer(1, 160, Relu()), Dropout(), Layer(160, 160, Relu()), Dropout(), Layer(160, 100, Relu()), Dropout(), Layer(100, 1, Relu())], optimizer=Adam(), callbacks=Early_Stopping(patience=10))
model.train(x, y, epochs=5000, learning_rate=5e-4, batch_size=32)
plt.plot(x, y)
plt.plot(x, model.predict(x))
plt.show()
plt.plot(np.arange(1, len(model.loss)+1), model.loss)
plt.show()
