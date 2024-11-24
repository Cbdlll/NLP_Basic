import torch
import torch.nn.utils as nn_utils
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, scheduler, epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
    
    def train(self):
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            
            train_loop = tqdm(self.train_loader, total= len(self.train_loader), desc=f"Training Epoch {epoch+1}/{self.epochs}")
            for sentences, labels in train_loop:
                sentences, labels = sentences.to(self.device), labels.to(self.device)

                # 前向传播
                loss = self.model.loss(sentences, labels)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                running_loss += loss.item()
                train_loop.set_postfix(loss=loss.item())
            self.scheduler.step(running_loss)
            
        torch.save(self.model.state_dict(), 'BiLSTM_CRF.pth')
        return running_loss