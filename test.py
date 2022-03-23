from re import A
from networks_and_layers import *
from matplotlib import pyplot as plt
import numpy as np


# theta = np.arange(-15, 15)
# theta = np.concatenate([theta, theta], axis=0)
# theta = np.concatenate([theta, theta], axis=0)
# x = 1000 * np.sin(theta) + theta - theta**3 + np.random.randn(theta.shape[0]) * 10
# y = theta**4 + np.random.randn(theta.shape[0]) * 100
# theta = theta.reshape(-1, 1)
# x = x.reshape(-1, 1)
# y = y.reshape(-1, 1)
# Y = np.concatenate((x, y), axis=1)

x = np.arange(-15, 15, 0.5)
# y = x**2
x = np.concatenate([x, x], axis=0)
x = np.concatenate([x, x], axis=0)
x = np.concatenate([x, x], axis=0)
y = x**2 + np.random.randn(x.shape[0])
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


model = Network(
    layers=[
        Layer(1, 15, Relu(), L1()), 
        Layer(15, 10, Relu(), L1()),
        Layer(10, 15, Relu(), L1()),
        Layer(15, 10, Relu(), L1()),
        Layer(10, 15, Relu(), L1()),
        Layer(15, 10, Relu(), L1()),
        Layer(10, 15, Relu(), L1()),
        Layer(15, 10, Relu(), L1()),
        Layer(10, 15, Relu(), L1()),
        Layer(15, 10, Relu(), L1()),
        Layer(10, 15, Relu(), L1()),
        Layer(15, 10, Relu(), L1()),
        Layer(10, 1, Relu(), L1())
    ],
    # loss=MAE(),
    optimizer=Adam(),
    callbacks=Early_Stopping(patience=10, min_delta=0.02),
)
model.train(x, y, epochs=5000, learning_rate=7e-2, batch_size=32)
# plt.scatter(x, y)
# plt.plot(
#     model.predict(np.arange(-15, 15).reshape(-1, 1))[:, 0],
#     model.predict(np.arange(-15, 15).reshape(-1, 1))[:, 1], color="orange"
# )
# plt.show()
# plt.scatter(theta, x)
# plt.plot(np.arange(-15, 15), model.predict(np.arange(-15, 15).reshape(-1, 1))[:, 0], color="orange")
# plt.show()
# plt.scatter(theta, y)
# plt.plot(np.arange(-15, 15), model.predict(np.arange(-15, 15).reshape(-1, 1))[:, 1], color="orange")
# plt.show()
# plt.plot(np.arange(1, len(model.loss) + 1), model.loss)
# plt.show()

plt.scatter(x, y)
plt.plot(np.arange(-20, 20).reshape(-1, 1), model.predict(np.arange(-20, 20).reshape(-1, 1)), color='orange')
plt.show()
plt.plot(np.arange(1, len(model.loss) + 1), model.loss)
plt.show()