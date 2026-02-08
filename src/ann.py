import torch 
import torch.nn as nn

import torch.utils.data as data_utils
from torchvision import datasets, transforms

class fashion(nn.Module):
    def __init(self,input_size):
        super(fashion, self).__init__()

        self.layer1=nn.Linear(128,64)
        self.layer2=nn.linear(128,32)
        self.layer3=nn.linear(32,10)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax()
    def forwards(self,input):
        input=self.relu(self.layer1(input))  
        input=self.relu(self.layer2(input))
        input=self.softmaz(self.layer3(input))

        return input
transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
dataset_obj=datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms)


print(dataset_obj)


