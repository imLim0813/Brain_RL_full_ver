# Import Library
import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from cnn_module import *
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

# GPU device
torch.cuda.set_device('cuda:1') if torch.cuda.is_available() else None

# Load data
target_data = np.load('../Data/target_pos.npy').astype('int')
print(target_data.shape)
cursor_data = np.load('../Data/cur_pos.npy').astype('int') # Just randomly choose.
print(cursor_data.shape)

frame_data = []
for i in range(target_data.shape[0]):
    frame_data.append([frame_data, to_frame(cur_x=cursor_data[i][0], cur_y=cursor_data[i][1],
                                            target_x=target_data[i][0], target_y=target_data[i][1])])
frame_data = np.array(frame_data)
print(frame_data.shape)

# Concatenate target and cursor
position_data = np.concatenate([target_data, cursor_data], axis=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(x=frame_data, y=position_data, test_size=0.2, random_state=42)

# train val split
X_train, X_val, y_train, y_val = train_test_split(x=X_train, y=y_train, test_size=0.2, random_state=42)

# Tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)

# Box Regression
model = models.alexnet(pretrained=True)


# Bounding Box : Alexnet ( except softmax layer ) + fully connected layer
class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        self.features1 = nn.Sequential(alexnet.features[:3])
        self.features2 = nn.Sequential(alexnet.features[3:6])
        self.features3 = nn.Sequential(alexnet.features[6:8])
        self.features4 = nn.Sequential(alexnet.features[8:10])
        self.features5 = nn.Sequential(alexnet.features[10:])
        self.avg = nn.Sequential(alexnet.avgpool)
        self.fc1 = nn.Sequential(alexnet.classifier[:3])
        self.fc2 = nn.Sequential(alexnet.classifier[3:6])

        self.bb = nn.Sequential(nn.BatchNorm1d(4096), nn.Linear(4096, 2048), nn.ReLU(),
                                nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 4))

    def forward(self, x):
        x = x.reshape((-1, 3, 227, 227))
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = self.avg(x)
        x = nn.Flatten()(x)

        x = self.fc1(x)
        x = self.fc2(x)
        return self.bb(x)


# Define optimizer
def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


# Training pipeline
def train_epocs(model, optimizer, train_dl, val_dl, va_list, epochs=10, C=1000):
    idx = 0
    va_list = va_list

    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0

        for x, y_bb in train_dl:
            batch = x.shape[0]
            x = x.cuda().float()
            y_bb = y_bb.cuda().float()
            out_bb = model(x)
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_bb / C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        val_loss = val_metrics(model, valid_dl, C)
        va_list.append(val_loss)
        print("Epoch : %d, train_loss %.3f val_loss %.3f " % (i+1, train_loss, val_loss))
        if val_loss == min(va_list):
            torch.save(model, './box_reg(alex).pt')
            print('model saved!')
            print('=' * 50)
        return sum_loss/total


# Validation pipeline
def val_metrics(model, valid_dl, C=1000):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x, y_bb in valid_dl:
        batch = x.shape[0]
        x = x.cuda().float()
        y_bb = y_bb.cuda().float()
        out_bb = model(x)
        loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_bb / C
        sum_loss += loss.item()
        total += batch
    return sum_loss/total


# Call the model
model = BB_model().cuda()


parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.000005)


train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_train, y_train)


train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_dl = DataLoader(val_ds, batch_size=64)
test_dl = DataLoader(test_ds, batch_size=64)


# Training
val_loss_list = []
train_epocs(model, optimizer, train_dl, valid_dl, epochs=100, va_list=val_loss_list)


# Validation
val_metrics(model, test_dl)

# Prediction
y_pred = model(X_test[0].cuda())
print('y_pred :', y_pred)
print('y_true :', y_test[0])
torch.save(model, './data/box_reg(alex).pt')