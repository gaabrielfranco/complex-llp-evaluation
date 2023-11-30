from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torchvision.models import resnet18

"""
This architecture results (first seed):

              precision    recall  f1-score   support

           0     0.8838    0.9003    0.8920      9329
           1     0.6566    0.6169    0.6361      2882

    accuracy                         0.8334     12211
   macro avg     0.7702    0.7586    0.7641     12211
weighted avg     0.8302    0.8334    0.8316     12211

"""

# Source: https://github.com/lucastassis/dllp/blob/main/net.py
class SimpleMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_sizes=(100,)):
        super(SimpleMLP, self).__init__()  
        self.layers = nn.ModuleList() 
        for size in hidden_layer_sizes:
            self.layers.append(nn.Linear(in_features, size))
            self.layers.append(nn.Dropout(0.5))
            self.layers.append(nn.ReLU())
            in_features = size
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))

    def forward(self, x): 
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    seed = [189395, 962432364, 832061813, 316313123, 1090792484,
            1041300646,  242592193,  634253792,  391077503, 2644570296, 
            1925621443, 3585833024,  530107055, 3338766924, 3029300153,
        2924454568, 1443523392, 2612919611, 2781981831, 3394369024,
            641017724,  626917272, 1164021890, 3439309091, 1066061666,
            411932339, 1446558659, 1448895932,  952198910, 3882231031]

    n_executions = 5

    base_dataset = "datasets-ci/cifar-10-grey-animal-vehicle.parquet"
    #base_dataset = "datasets-ci/adult.parquet"

    # Reading X, y (base dataset) and bags (dataset)
    df = pd.read_parquet(base_dataset)
    X = df.drop(["y"], axis=1).values
    y = df["y"].values
    y = y.reshape(-1)

    # CIFAR-10
    X = X.reshape(-1, 1, 32, 32)

    for execution in range(n_executions):
        train_index, test_index = next(ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed[execution]).split(X))

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # Training SimpleMLP
        device = "mps"
        n_epochs = 10
        #model = SimpleMLP(X_train.shape[1], 2, hidden_layer_sizes=(1000,))
        # Restnet18
        model = resnet18(weights="IMAGENET1K_V1")
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Linear(model.fc.in_features, 2)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Convert to tensor (float32)
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()

        # Create a dataloder with X, y
        data_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=64, shuffle=False, num_workers=8)

        model.train()
        with tqdm(range(n_epochs), desc='Training model', unit='epoch') as tepoch:
            for i in tepoch:
                losses = []
                for X, y in data_loader:
                    # prepare bag data
                    X, y = X.to(device), y.to(device)
                    # compute outputs
                    outputs = model(X)
                    # compute loss and backprop
                    loss = criterion(outputs, y)
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                train_loss = np.mean(losses)
                print("[Epoch: %d/%d] train loss: %.4f" % (i + 1, n_epochs, train_loss))

        # Testing SimpleMLP
        test_loader = torch.utils.data.DataLoader(
            X_test,
            batch_size=64,
            shuffle=False,
            num_workers=8,
        )
        y_pred = []

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = batch.to(device)
                pred = model(x)
                y_pred += pred.argmax(dim=1).cpu().tolist()

        y_pred = np.array(y_pred).reshape(-1)
        print(classification_report(y_test, y_pred, digits=4))
        print("F1-score: %.4f" % f1_score(y_test, y_pred))
        exit()


