import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# Define the neural network architecture

def plot_classes_preds(outputs, images, labels):
    # Get the predicted classes for the batch
    _, predicted = torch.max(outputs, 1)

    # Prepare the grid of images and their corresponding labels and predictions
    fig = plt.figure(figsize=(12, 12))
    for i in range(min(25, images.shape[0])):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        ax.set_title(
            f"Label: {labels[i].cpu()}, Prediction: {predicted[i].cpu()}",
            color=("green" if predicted[i] == labels[i] else "red"))
        ax.axis('off')

    return fig


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Load and transform the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
net = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Initialize TensorBoard writer
writer = SummaryWriter()

# Train the model
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training loss to TensorBoard
        writer.add_scalar(
            'Training Loss', loss.item(),
            epoch * total_steps + i)

        if (i + 1) % 100 == 0:
            writer.add_images('Input Images', images,
                              global_step=epoch * total_steps + i)
            writer.add_figure('Predictions', plot_classes_preds(
                outputs, images, labels), global_step=epoch * total_steps + i)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")

    # Test the model on the test dataset
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        # Log test
        writer.add_scalar('Test Accuracy', accuracy, epoch)

        print(
            f"Test Accuracy of the model on the {total} test images: {accuracy:.2f} %")

    # Switch the model back to training mode
    net.train()
writer.close()
