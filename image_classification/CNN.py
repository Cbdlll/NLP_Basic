import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

#define Hyperparameters
image_size = 256  
epochs = 10
num_classes = 2
batch_size = 32
learning_rate = 0.001


#data preprocess
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
])

#load data
def load_data(path, transform):
    train_dataset = datasets.ImageFolder(f"{path}/train", transform=transform)
    val_dataset = datasets.ImageFolder(f"{path}/val", transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader

#define model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   #256x256x3 -> 256x256x32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  #256x256x32 -> 256x256x64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   #256x256x64 -> 128x128x64
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  #128x128x64 -> 128x128x64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)   #128x128x64 -> 32x32x64
        
        self.output = nn.Sequential(
            nn.Linear(32*32*64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        
    def forward(self, x):  
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        
        x = x.view(-1, 32*32*64)
        x = self.output(x)
        return x
        
#train model
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer ,num_classes, epochs, device):
    best_acc = 0.0
    best_class_acc = {}
    
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_loop = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}", ncols=150)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            train_loop.set_postfix(loss=running_loss/total, accuracy=correct/total*100)
            
        val_acc, class_acc = val_model_with_classes(model, val_dataloader, num_classes, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_class_acc = class_acc
            best_model_state = model.state_dict()
        
    torch.save(best_model_state, 'CNN_classifier.pth')
    
    print(f"Total Val Acc: {best_acc:.2f}%")
    for i in range(num_classes):
        print(f"Class {i} Accuracy: {best_class_acc[i]:.2f}%")
            

    
#val model
def val_model_with_classes(model, val_dataloader, num_classes, device):
    correct = 0
    total = 0
    
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            for i in range(num_classes):
                correct_per_class[i] += ((predicted == i) & (labels == i)).sum().item()
                total_per_class[i] += (labels == i).sum().item()
                
    class_acc = {}
    for i in range(num_classes):
        if total_per_class[i] > 0:
            class_acc[i] = correct_per_class[i] / total_per_class[i] * 100
        else:
            class_acc[i] = 0.0
            
    val_acc = correct / total * 100

    return val_acc, class_acc
      
#Training
def Train(model, criterion, optimizer, data_path, num_classes, epochs, device):
    train_dataloader, val_dataloader = load_data(data_path, transform)
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_classes, epochs, device)
    

if __name__ == '__main__':
    data_path = 'new_data'
    
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Train(model, criterion, optimizer, data_path, num_classes, epochs, device)
    