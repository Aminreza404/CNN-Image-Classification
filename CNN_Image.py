import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
import torchvision.transforms as transforms 
import torch 
import torch.nn as nn 
import torch.optim as optim
from tqdm.auto import tqdm 


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

print("CIFAR-10 dataset downloaded and loaded successfully!")

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

dataiter = iter(trainloader)
images, labels = next(dataiter)

plt.figure(figsize=(8,5))
for i in range(8):
    plt.subplot(2,4,i+1)

    img = images[i]
    img = img * 0.5 + 0.5        # unnormalize
    img = img.permute(1, 2, 0)  # C,H,W → H,W,C

    plt.imshow(img)
    plt.title(classes[labels[i]])
    plt.axis('off')

plt.show()


class CNNModel(nn.Module): 
    def __init__(self): 
        super().__init__() 

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(512, 10) 


    def forward(self, x): 
        x = torch.relu(self.conv1(x)) 
        x = torch.max_pool2d(x, 2) 
        x = torch.relu(self.conv2(x)) 
        x = torch.max_pool2d(x, 2) 
        x = torch.relu(self.conv3(x)) 
        x = torch.max_pool2d(x, 2) 
        x = x.view(-1, 128 * 4 * 4) 
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

num_epochs = 10

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_losses.append(running_loss / len(trainloader))
    train_accuracies.append(100 * correct_train / total_train)
    
    model.eval()
    running_loss = 0.0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    test_losses.append(running_loss / len(testloader))
    test_accuracies.append(100 * correct_test / total_test)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, "
          f"Train Accuracy: {train_accuracies[-1]:.2f}%, Test Accuracy: {test_accuracies[-1]:.2f}%")


plt.rc('text', usetex = True)
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label = r"$train$ $losses$") 
plt.plot(range(1, num_epochs+1), test_losses, label = r"$test$ $losses$") 
plt.legend(fontsize = 18) 
plt.ylabel(r"$Loss$ $function$", fontsize = 18) 
plt.xlabel(r"$Epoch$", fontsize = 18) 
plt.tick_params(direction = "in", labelsize = 18)


plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_accuracies, label=r'$Train$ $Accuracy$')
plt.plot(range(1, num_epochs+1), test_accuracies, label=r'$Test$ $Accuracy$')
plt.xlabel(r'$Epoch$', fontsize = 18)
plt.ylabel(r'$Accuracy$', fontsize = 18)
plt.tick_params(direction = "in", labelsize = 18)
plt.legend(fontsize = 18)
