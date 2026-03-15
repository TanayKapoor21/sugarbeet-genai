import torch
import random
import numpy as np

class AdvancedHSIDataset(torch.utils.data.Dataset):

    def __init__(self,X,y,augment=True):

        self.X = torch.tensor(X).permute(0,3,1,2).float()
        self.y = torch.tensor(y).long()

        self.augment=augment

    def spectral_noise(self,x):

        noise = torch.randn_like(x)*0.02
        return x + noise

    def spectral_shift(self,x):

        shift = random.randint(-2,2)
        return torch.roll(x,shift,dims=0)

    def mixup(self,x,y):

        lam = np.random.beta(0.4,0.4)

        idx = random.randint(0,len(self.X)-1)

        x2 = self.X[idx]
        y2 = self.y[idx]

        x = lam*x + (1-lam)*x2

        return x,y

    def __len__(self):

        return len(self.X)

    def __getitem__(self,idx):

        x = self.X[idx]
        y = self.y[idx]

        if self.augment:

            if random.random()<0.5:
                x = self.spectral_noise(x)
            if random.random()<0.5:
                x = self.spectral_shift(x)
            if random.random() < 0.5:
                x = x + torch.randn_like(x) * 0.05
        return x,y