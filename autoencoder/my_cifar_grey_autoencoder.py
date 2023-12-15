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
class CIFARAE(nn.Module):
    def __init__(self):
        super(CIFARAE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            #nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.Conv2d(1, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
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
            #nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.ConvTranspose2d(12, 1, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
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

    # download data and create dataloader 


    df = pd.read_parquet("../base-datasets/cifar-10-grey.parquet")
    df = df.sample(frac=1, random_state=random_state)
    df.label = df.label.apply(lambda x: 0 if x in [0, 1, 8, 9] else 1)
    X = deepcopy(df.drop(columns=["label"]).values)
    y = deepcopy(df["label"].values)
    X = MinMaxScaler().fit_transform(X)
    trainset = torch.from_numpy(X).float()

    # transform = transforms.Compose([transforms.ToTensor(),
    #                             transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)) ]
    #                         )
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False) # test loader using same train data (but batch_size=1) to generate latent vectors

    # define model and loss
    model = CIFARAE()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # # train model
    # num_epochs = 100
    # model.train()
    # for epoch in range(num_epochs):
    #     for data in tqdm(train_loader):
    #         #img, _ = data
    #         img = data
    #         img = img.reshape(-1, 1, 32, 32)
    #         encoded, decoded = model(img)
    #         optimizer.zero_grad()
    #         loss = criterion(decoded, img)
    #         loss.backward()
    #         optimizer.step()
    #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    # torch.save(model.state_dict(), 'weights-cifar-10-grey-animal-vehicle.pth') # save weights

    # generate latent vectors and save to file
    model.load_state_dict(torch.load('weights-cifar-10-grey-animal-vehicle.pth'))
    model.eval()
    with torch.no_grad():
        latent_data = torch.concat([model(data.reshape(1, 32, 32))[0].flatten().reshape(1, -1) for data in test_loader]).numpy()

    def compute_proportions(bags, y):
        """Compute the proportions for each bag given.
        Parameters
        ----------
        bags : {array-like}
        y : {array-like}

        Returns
        -------
        proportions : {array}
            An array of type np.float
        """
        num_bags = int(max(bags)) + 1
        proportions = np.empty(num_bags, dtype=float)
        for i in range(num_bags):
            bag = np.where(bags == i)[0]
            proportions[i] = np.count_nonzero(y[bag] == 1) / len(bag)
        return proportions

    n_clusters = 50

    clusters = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state).fit_predict(latent_data)
    bags = clusters
    proportions_found, bags_size_found = compute_proportions(clusters, y), np.bincount(clusters)

    base_dataset = "cifar-10-grey-animal-vehicle"
    llp_variant = "intermediate"
    n_bags_type = "massive"
    bags_size_type = proportions_type = "fol-clust"
    clustering_method = "kmeans-autoencoder"
    

    # Saving only the bags (saving space)
    df = pd.DataFrame(bags, columns=["bag"], dtype=int)

    filename = "{}-{}-{}-{}-{}-cluster-{}-{}.parquet".format(base_dataset, llp_variant, n_bags_type, \
                        bags_size_type, 
                        proportions_type, \
                        clustering_method, \
                        n_clusters)
    df.to_parquet(filename, index=False)
    print("Dataset {} generated".format(filename))
    print("Proportions: {}".format(proportions_found))
    print("Bag sizes: {}".format(bags_size_found))
    print("\n------------------------\n")


    

    # X_tensors = np.array([trainset[i][0].numpy() for i in range(0, len(trainset))])
    # y_tensors = np.array([trainset[i][1] for i in range(0, len(trainset))])

    # print(X_tensors.shape, y_tensors.shape, latent_data.shape)
        
    # # save train data
    # with open(f'cifar_latent.npy', 'wb') as f:
    #     np.save(f, X_tensors, allow_pickle=False)
    #     np.save(f, y_tensors, allow_pickle=False)
    #     np.save(f, latent_data, allow_pickle=False)

    # # visualize some examples
    # def imshow(img):
    #     npimg = img.cpu().numpy()
    #     plt.axis('off')
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()

    # model.load_state_dict(torch.load('weights_cifar.pth'))
    # model.eval()
    # with torch.no_grad():
    #     for data, y in test_loader:
    #         encoded, output = model(data)
    #         grid = torchvision.utils.make_grid(torch.concat([data, output]))
    #         imshow(grid)

if __name__ == '__main__':
    main()