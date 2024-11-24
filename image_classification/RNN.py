import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

#define Hyperparameters
input_size = 128
hidden_size = 128
num_layers = 2
num_classes = 2
batch_size = 32
epochs = 50
learning_rate = 0.001

#define model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        rnn_out, _ = self.rnn(x, h0)
        rnn_out = rnn_out[:, -1, :] #(batch_size, sequence_length, hidden_size)
        out = self.fc(rnn_out)
        return out

#data preprocess
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

#load data
def load_data(data_path, transform):
    train_dataset = datasets.ImageFolder(f"{data_path}/train", transform=transform)
    val_dataset = datasets.ImageFolder(f"{data_path}/val", transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader

#train model
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_classes, epochs, device):
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
            
            inputs = inputs.view(inputs.size(0), -1, input_size)
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
                
    torch.save(best_model_state, 'RNN_classifier.pth')
    print(f"Total Val Acc: {best_acc:.2f}%")
    for i in range(num_classes):
        print(f"Class {i} Accuracy: {best_class_acc[i]:.2f}%")
            


#val model
def val_model_with_classes(model, val_dataloader, num_classes, device):
    correct = 0
    total = 0
    
    correct_per_class = [0] * num_classes
    total_per_class = [0] *num_classes
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            inputs = inputs.view(inputs.size(0), -1, input_size)
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
def Train(model, criterion, optimizer, data_path, num_classes, epochs, device=None):
    train_dataloader, val_dataloader = load_data(data_path, transform)
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_classes, epochs, device)
                
if __name__ == '__main__':
    data_path = 'new_data'
    
    model = RNN(input_size, hidden_size, num_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Train(model, criterion, optimizer, data_path, num_classes, epochs, device=device)