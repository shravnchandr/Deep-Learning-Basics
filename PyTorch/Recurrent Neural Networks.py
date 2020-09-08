# Importing Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Checking for GPU(cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting Hyperparameters
num_epochs = 10
num_classes = 10
batch_size = 1000
learning_rate = 0.001
hidden_size = 128
num_layers = 3
input_size = 28
sequence_len = 28

# Downloading FMNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root='dataset', train=True, 
                                                    transform=transforms.ToTensor(),
                                                    download=True)
val_dataset = torchvision.datasets.FashionMNIST(root='dataset', train=False, 
                                                    transform=transforms.ToTensor())

# Data Loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=10):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=False)
        self.func = nn.Sequential(
            nn.Linear(hidden_size *1, 32), nn.ReLU(),    # x2 for Bi-Directional RNN
            nn.Linear(32, num_classes))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers *1, x.size(0), self.hidden_size).to(device)    # x2 for Bi-Directional RNN
        rnn_out, _ = self.rnn(x, h0)
        out = self.func(rnn_out[:, -1, :])
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=10):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False)
        self.func = nn.Sequential(
            nn.Linear(hidden_size *1, 32), nn.ReLU(),    # x2 for Bi-Directional LSTM
            nn.Linear(32, num_classes))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers *1, x.size(0), self.hidden_size).to(device)    # x2 for Bi-Directional LSTM
        c0 = torch.zeros(self.num_layers *1, x.size(0), self.hidden_size).to(device)    # x2 for Bi-Directional LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.func(lstm_out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=10):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=False)
        self.func = nn.Sequential(
            nn.Linear(hidden_size *1, 32), nn.ReLU(),    # x2 for Bi-Directional GRU
            nn.Linear(32, num_classes))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers *1, x.size(0), self.hidden_size).to(device)    # x2 for Bi-Directional GRU
        gru_out, _ = self.gru(x, h0)
        out = self.func(gru_out[:, -1, :])
        return out

model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss Function & Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the Model
for epoch in range(num_epochs):
    for step, (image, label) in enumerate(train_loader):
        image = image.reshape(-1, sequence_len, input_size).to(device)
        label = label.to(device)

        output = model(image)
        loss = loss_func(output, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Validating the Model
        if (step+1) % 60 == 0:
            correct_prediction = 0
            total = 0
            for val_image, val_label in val_loader:
                val_image = val_image.reshape(-1, sequence_len, input_size).to(device)
                val_label = val_label.to(device)
                output = model(val_image)

                _, prediction = torch.max(output.data, 1)
                
                total += val_label.size(0)
                correct_prediction += (prediction == val_label).sum().item()

            val_accuracy = (correct_prediction / total) * 100
            print('Epoch [{}/{}], Loss: {:.4f}, Validation Accuracy: {} %'
                   .format(epoch+1, num_epochs, loss.item(), val_accuracy))

# Saving the Model Checkpoint
torch.save(model.state_dict(), 'model.ckpt')
