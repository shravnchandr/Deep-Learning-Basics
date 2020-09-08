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

# Downloading FMNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root='dataset', train=True, 
                                                    transform=transforms.ToTensor(),
                                                    download=True)
val_dataset = torchvision.datasets.FashionMNIST(root='dataset', train=False, 
                                                    transform=transforms.ToTensor())

# Data Loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

'''
CNN Layout
1.  Feature Map of 16 channels, with Kernel Size of (5x5) and Padding of 2. Then followed by MaxPooling with Kernel Size of (2x2) and Stride of 2.
2.  Feature Map of 16 channels, with Kernel Size of (3x2).
3.  Feature Map of 32 channels, with Kernel Size of (5x5) and Padding of 2. Then followed by MaxPooling with Kernel Size of (2x2) and Stride of 2.

ANN Layout
Single Hidden Layer of 32 Units with ReLU Activation Function.
'''


# CNN Option 1, each filter and activation function have to be called individually.
'''
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.batchN1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cnn2 = nn.Conv2d(16, 16, kernel_size=3)
        self.batchN2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        self.cnn3 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.batchN3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.func1 = nn.Linear(6*6*32, 32)
        self.relu4 = nn.ReLU()
        self.func2 = nn.Linear(32, num_classes)

    def forward(self, x):
        conv1_out = self.maxpool1(self.relu1(self.batchN1(self.cnn1(x))))
        conv2_out = self.relu2(self.batchN2(self.cnn2(conv1_out)))
        conv3_out = self.maxpool3(self.relu3(self.batchN3(self.cnn3(conv2_out))))

        func_input = conv3_out.reshape(conv3_out.size(0), -1)
        hidden_out = self.relu4(self.func1(func_input))
        out = self.func2(hidden_out)
        return out
'''


# CNN option 2, required filter and activation function is defined inside the Sequential, and hence no need to call each one individually.
'''
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16), nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.func = nn.Sequential(
            nn.Linear(6*6*32, 32), nn.ReLU(),
            nn.Linear(32, num_classes))

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)

        func_input = conv3_out.reshape(conv3_out.size(0), -1)
        out = self.func(func_input)
        return out
'''

model = CNN(num_classes).to(device)

# Loss Function & Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the Model
for epoch in range(num_epochs):
    for step, (image, label) in enumerate(train_loader):
        image = image.to(device)
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
                val_image = val_image.to(device)
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
