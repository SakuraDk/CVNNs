import torch
from complex_torch import *
import complex_nn as cvnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use: " + str(device))

# set data
inputs = [Complex(x=torch.Tensor([[-1], [-1]]), y=torch.zeros(2, 1)),
          Complex(x=torch.Tensor([[-1], [1]]), y=torch.zeros(2, 1)),
          Complex(x=torch.Tensor([[1], [-1]]), y=torch.zeros(2, 1)),
          Complex(x=torch.Tensor([[1], [1]]), y=torch.zeros(2, 1))]

targets = [0, 1, 1, 0]  # XOR

# Hyper Parameters
n_epoch = 20
n_category = 2
n_part = 2

# set model
model = cvnn.ComplexNN([2, 10, 10, 1], device=device)

# train model
for epoch in range(n_epoch):
    for i in range(4):
        z = inputs[i].to(device)
        model(z)
        model.train(targets[i], n_category, n_part=n_part)

# test model
for i in range(4):
    z = inputs[i].to(device)
    printComp(z)
    output = model(z)
    print(int(output.imag < 0))

"""
[2, 3, 1]
n_part = 1
0.11
n_part = 2
0.18

[2, 10, 1]
n_part = 1
0.85
n_part = 2
0.88

[2, 20, 1]
n_part = 1
0.91
n_part = 2
0.90

[2, 50, 1]
n_part = 1
0.78
n_part = 2
0.69

[2, 5, 5, 1]
n_part = 1
0.76
n_part = 2
0.83

[2, 5, 10, 1]
n_part = 1
0.91
n_part = 2
0.91

[2, 8, 8, 1]
n_part = 1
0.95
n_part = 2
0.94

[2, 10, 10, 1]
n_part = 1
0.99
n_part = 2
1.00
"""