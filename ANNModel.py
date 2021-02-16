import torch
import torch.nn as nn
import torch.nn.functional as nf

class ANNModel(nn.Module):
  def __init__(self,input_features=4,hidden1=8,hidden2=16,out_features=2):
    super().__init__()
    self.fc1=nn.Linear(input_features,hidden1)
    self.fc2=nn.Linear(hidden1,hidden2)
    self.out=nn.Linear(hidden2,out_features)
  def forward(self,x):
    x=nf.relu(self.fc1(x))
    x=nf.relu(self.fc2(x))
    x=nf.softmax(self.out(x))
    return x
