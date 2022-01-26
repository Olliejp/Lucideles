import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class Engine:
    def __init__(self, model, optimizer, scheduler, device=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.device = device
    
    @staticmethod
    def loss_fn(y, yhat):
        return nn.MSELoss()(y, yhat)
    
    def train(self, dataloader):
        self.model.train()
        final_loss = 0
        for xb, yb in dataloader:
            xb = xb.float().to(DEVICE)
            yb = yb.float().to(DEVICE)
            self.optimizer.zero_grad()
            y_hat = self.model(xb)
            loss = self.loss_fn(yb, y_hat)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        #loss_per_epoch = final_loss / len(dataloader)
        #self.scheduler.step(loss_per_epoch)
        print(f"learning rate: {self.optimizer.param_groups[0]['lr']}")
        return final_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        final_loss = 0
        for xb, yb in dataloader:
            xb = xb.float().to(DEVICE)
            yb = yb.float().to(DEVICE)
            y_hat = self.model(xb.float())
            loss = self.loss_fn(yb, y_hat)
            final_loss += loss.item()
        self.scheduler.step(final_loss / len(dataloader))
        return final_loss / len(dataloader)


class Linear(nn.Module):
    def __init__(self, n_features, n_targets):
        super().__init__()
        self.model = nn.Linear(n_features, n_targets)
    
    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, n_features, 
                n_targets, hidden_size, 
                n_layers, dropout, batch_norm = False):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(n_features, hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, n_targets))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


                

