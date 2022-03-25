from re import A
from networks_and_layers import *
from matplotlib import pyplot as plt
import numpy as np


# theta = np.arange(-15, 15, 0.1)
# theta = np.concatenate([theta, theta], axis=0)
# theta = np.concatenate([theta, theta], axis=0)
# x = theta * np.sin(theta) + np.random.randn(theta.shape[0])/10
# y = theta * np.cos(theta) + np.random.randn(theta.shape[0])/10
# theta = theta.reshape(-1, 1)
# x = x.reshape(-1, 1)
# y = y.reshape(-1, 1)
# Y = np.concatenate((x, y), axis=1)

x = np.arange(-100, 100, 0.5)
x = np.concatenate([x, x], axis=0)
x = np.concatenate([x, x], axis=0)
x = np.concatenate([x, x], axis=0)
y = x**2 + np.random.randn(x.shape[0])*300 + 60
# y = (x/10)*np.cos(x/10) + x*np.random.randn(x.shape[0])/25 + 60
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


model = Network(
    layers=[
        Layer(1, 200, Relu(), regularizer=L1(0.000001)),
        Layer(200, 1),
    ],
    # loss=MAE(),
    optimizer=Adam(),
    callbacks=Early_Stopping(patience=15, min_delta=0),
)


def lr(epoch):
    return 9e-3 * (1 - (epoch / 10000) ** 3)


model.train(x, y, epochs=5000, learning_rate=lr, batch_size=32)

# plt.scatter(x, y)
# plt.plot(
#     model.predict(np.arange(-15, 15, 0.1).reshape(-1, 1))[:, 0],
#     model.predict(np.arange(-15, 15, 0.1).reshape(-1, 1))[:, 1],
#     color="orange",
# )
# plt.show()

# plt.scatter(x, y)
# plt.plot(
#     model.predict(np.arange(-15, 15).reshape(-1, 1))[:, 0],
#     model.predict(np.arange(-15, 15).reshape(-1, 1))[:, 1],
#     color="orange",
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
plt.plot(
    np.arange(-100, 100).reshape(-1, 1),
    model.predict(np.arange(-100, 100).reshape(-1, 1)),
    color="orange",
)
plt.show()
plt.plot(np.arange(1, len(model.loss) + 1), model.loss)
plt.show()
# plt.plot(np.arange(1, len(model.loss) + 1), lr(np.arange(1, len(model.loss) + 1)), color='green')
# plt.show()
