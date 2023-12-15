from copy import deepcopy
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

# from: https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py
class SVHNAE(nn.Module):
    def __init__(self):
        super(SVHNAE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def main():
    # parameters
    lr = 0.001
    random_state=6738921

    CPU=False
    device = "cpu" if CPU else torch.device("mps")
    print("Device is :: {}".format(device))

    # download data and create dataloader 

    df = pd.read_parquet("../base-datasets/svhn.parquet")
    df = df.sample(frac=1, random_state=random_state)
    X = deepcopy(df.drop(columns=["label"]).values)
    y = deepcopy(df["label"].values)
    X = MinMaxScaler().fit_transform(X)
    trainset = torch.from_numpy(X).float()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False) # test loader using same train data (but batch_size=1) to generate latent vectors

    # define model and loss (to run on mps device)
    model = SVHNAE()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # train model
    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        for data in tqdm(train_loader):
            #img, _ = data
            img = data
            img = img.reshape(-1, 3, 32, 32)
            img = img.to(device)
            encoded, decoded = model(img)
            optimizer.zero_grad()
            loss = criterion(decoded, img)
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    torch.save(model.state_dict(), 'weights-svhn.pth') # save weights

if __name__ == '__main__':
    main()