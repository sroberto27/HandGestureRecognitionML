import itertools
import torch
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
#from d2l import torch as d2l

# Define model architecture
class HandGestureCNN(nn.Module):
    def __init__(self):
        super(HandGestureCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjust input size to match output size of last convolutional layer
        self.fc1 = nn.Linear(50688, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        batch_size = x.size(0) # Get the batch size
        x = x.view(batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#create model
# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandGestureCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load dataset using DataLoader
train_transforms = transforms.Compose([
    transforms.Resize((176, 144)),
    transforms.RandomCrop((176, 144)),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder('data/train', transform=train_transforms)
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(len(train_dataset))
# Train model
for epoch in range(10):
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        print(f"Epoch {epoch+1}:"f"  Batch {i+1}:")
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Export the model to ONNX format
dummy_input = torch.randn(1, 1, 176, 144)
output_file = "gesture_cnn.onnx"
torch.onnx.export(model, dummy_input, output_file)

# Save model in PyTorch's .pt format
torch.save(model.state_dict(), 'hand_gesture_model.pt')


# Test model
model.eval() # set model to evaluation mode
test_transforms = transforms.Compose([
    transforms.Resize((176, 144)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_dataset = datasets.ImageFolder('data/test', transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

# Evaluate the model GRAPHICALLY
model.eval() # set model to evaluation mode
test_transforms = transforms.Compose([
    transforms.Resize((176, 144)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_dataset = datasets.ImageFolder('data/test2', transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
classes = ['fist', 'open Hand', 'thumps down', 'thumps up']
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        for i in range(data.size(0)):
            plt.imshow(data[i][0], cmap='gray')
            plt.title(f"True label: {classes[target[i]]}, Predicted label: {classes[predicted[i]]}")
            plt.show()


# Evaluate the model graph
model.eval() # set model to evaluation mode
test_transforms = transforms.Compose([
    transforms.Resize((176, 144)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_dataset = datasets.ImageFolder('data/test', transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
y_true = []
y_pred = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        y_true += target.tolist()
        y_pred += predicted.tolist()

# Plot confusion matrix
classes = ['gesture1', 'gesture2', 'gesture3', 'gesture4']
cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = '.2f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
