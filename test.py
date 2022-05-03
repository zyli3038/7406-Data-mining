import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random

data = np.array(pd.read_csv("./data.csv", sep=","))
test = np.array(pd.read_csv("./test.csv", sep=","))
# X1 = np.array(data["X1"])
# X2 = np.array(data["X2"])
# mu = np.array(data["muhat"])
# V = np.array(data["Vhat"])


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        out = self.predict(x)
        return out


net = Net(2, 64, 1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

N = range(10000)
n1 = 7000
n1 = random.sample(N, n1)
n2 = list(set(N).difference(set(n1)))

X = data[:, 1:3]
Y = data[:, 3].reshape(10000, 1)

X_train = X[n1, :]
Y_train = Y[n1]

X_test = X[n2, :]
Y_test = Y[n2]

x_data = torch.tensor(X_train, dtype=torch.float32)
y_data = torch.tensor(Y_train, dtype=torch.float32)

x_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(Y_test, dtype=torch.float32)

plt.ion()
epoch = 200001
loss_ls = []
for step in range(epoch):
    predict_y = net(x_data)
    loss = loss_func(predict_y, y_data)
    loss_ls.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 2000 == 0:
        print("step {} | loss: {}.".format(step, loss))
        plt.cla()
        ax = plt.subplot(111, projection="3d")
        ax.scatter(x_data[:, [0]], x_data[:, [1]], y_data, c='g')
        ax.scatter(x_data[:, [0]], x_data[:, [1]], predict_y.data.numpy(), c='r')
        plt.pause(0.1)

plt.ioff()
plt.show()


y_train_pred = net.forward(x_data)
print("mse for train: ", torch.mean((y_train_pred - y_data)**2))

y_test_pred = net.forward(x_test)
print("mse for test: ", torch.mean((y_test_pred - y_test)**2))

test_input = torch.tensor(test, dtype=torch.float32)
test_output = net.forward(test_input).data.numpy()
np.savetxt("./mu.csv", test_output, delimiter=",")

