import numpy
import pandas
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = torchvision.datasets.FashionMNIST(root='dataset', train=True, 
                                                    transform=transforms.ToTensor(),
                                                    download=True)
val_dataset = torchvision.datasets.FashionMNIST(root='dataset', train=False, 
                                                    transform=transforms.ToTensor())

y_train = numpy.array(training_dataset.iloc[:, 0])
x_train = numpy.array(training_dataset.drop(training_dataset.columns[0], axis= 'columns')) / 255


# No need to One-Hot Encode the labels
'''output_layer = [[0 for _ in range(10)] for _ in range(y_train.shape[0])]
for i in range(y_train.shape[0]):
    output_layer[i][y_train[i]] = 1
y_train = numpy.array([numpy.array(y_i) for y_i in output_layer])'''

x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

train_data = TensorDataset(x_train_tensor, y_train_tensor)

y_validate = numpy.array(val_dataset.iloc[:, 0])
x_validate = numpy.array(val_dataset.drop(val_dataset.columns[0], axis= 'columns')) / 255

'''output_layer = [[0 for _ in range(10)] for _ in range(y_validate.shape[0])]
for i in range(y_validate.shape[0]):
    output_layer[i][y_validate[i]] = 1
y_validate = numpy.array([numpy.array(y_i) for y_i in output_layer])'''

x_validate_tensor = torch.from_numpy(x_validate).float()
y_validate_tensor = torch.from_numpy(y_validate).float()

validate_data = TensorDataset(x_validate_tensor, y_validate_tensor)

weights1 = torch.randn((784, 32), requires_grad= True, dtype= torch.float, device= device)
weights2 = torch.randn((32, 16), requires_grad= True, dtype= torch.float, device= device) 
weights3 = torch.randn((16, 10), requires_grad= True, dtype= torch.float, device= device) 

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD([weights1, weights2], lr= 0.01)

y_train_tensor = y_train_tensor.long()
y_validate_tensor = y_validate_tensor.long()

losses = []

for epoch in range(256):
    hidden1 = torch.mm(x_train_tensor, weights1)
    hidden2 = torch.mm(hidden1, weights2)
    yhat = torch.mm(hidden2, weights3)

    loss = loss_func(yhat, y_train_tensor)
    loss.backward()

    losses.append(loss)

    optimizer.step()
    optimizer.zero_grad()

print(losses[-1])

hidden1_temp = torch.mm(x_validate_tensor, weights1)
hidden2_temp = torch.mm(hidden1_temp, weights2)
yhat_temp = torch.mm(hidden2_temp, weights3)
loss_temp = loss_func(yhat_temp, y_validate_tensor)

temp_y_val = y_validate_tensor.numpy()
_, val_pre = yhat_temp.max(1)
val_pre = val_pre.numpy()

from sklearn.metrics import accuracy_score
print(accuracy_score(temp_y_val, val_pre))

print(loss_temp)
