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
class CIFARGreyAE(nn.Module):
    def __init__(self):
        super(CIFARGreyAE, self).__init__()
        # Input size: [batch, 1, 32, 32]
        # Output size: [batch, 1, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 1, 4, stride=2, padding=1),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    df = pd.read_parquet("../base-datasets/cifar-10-grey.parquet")
    df = df.sample(frac=1, random_state=random_state) # Shuffle
    df.label = df.label.apply(lambda x: 0 if x in [0, 1, 8, 9] else 1)
    X = deepcopy(df.drop(columns=["label"]).values)
    y = deepcopy(df["label"].values)
    X = MinMaxScaler().fit_transform(X)
    trainset = torch.from_numpy(X).float()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False, num_workers=2)

    # define model and loss
    model = CIFARGreyAE()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # # train model
    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        for data in tqdm(train_loader):
            img = data
            img = img.reshape(-1, 1, 32, 32).to(device)
            encoded, decoded = model(img)
            optimizer.zero_grad()
            loss = criterion(decoded, img)
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    torch.save(model.state_dict(), 'weights-cifar-10-grey-animal-vehicle.pth') # save weights

if __name__ == '__main__':
    main()