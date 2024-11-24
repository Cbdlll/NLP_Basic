import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

#define Hyperparameters
image_size = 150
batch_size = 32
learning_rate = 0.001
num_classes = 2
epochs = 10

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

#define Model
class DNN(nn.Module):
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

#train model

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_classes=2, epochs=10, device=None):  
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
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
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
def validate_model(model, val_loader, device=None):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total * 100

#val model with each class_acc
def val_model_with_classes(model, val_dataloader, num_classes, device=None):
    model.eval()
    correct = 0
    total = 0
    
    # 初始化每个类别的正确预测和总计数
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算每个类别的正确预测和总数
            for i in range(num_classes):
                correct_per_class[i] += ((predicted == i) & (labels == i)).sum().item()
                total_per_class[i] += (labels == i).sum().item()

    # 计算每个类别的准确率
    class_acc = {}
    for i in range(num_classes):
        if total_per_class[i] > 0:
            class_acc[i] = correct_per_class[i] / total_per_class[i] * 100
        else:
            class_acc[i] = 0.0  # 如果某个类别没有数据，精度设置为0

    overall_acc = correct / total * 100

    # 打印各类别的准确率
    # print(f"Overall Accuracy: {overall_acc:.2f}%")
    # for i in range(num_classes):
    #     print(f"Class {i} Accuracy: {class_acc[i]:.2f}%")
    
    return overall_acc, class_acc

#Training
def Train(model, criterion, optimizer, data_path, num_classes, epochs, device):
    train_dataloader, val_dataloader = load_data(data_path, transform)
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_classes, epochs, device)


if __name__ == '__main__':
    data_path = 'new_data'
 
    input_size = image_size * image_size * 3
    model = DNN(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Train(model, criterion, optimizer, data_path, num_classes, epochs, device)