import numpy as np
import scipy.io
train = scipy.io.loadmat('train_32x32.mat')
test = scipy.io.loadmat('test_32x32.mat')

X = train['X']
y = train['y']
X_test = test['X']
y_test = test['y']

y = y.reshape(-1)
y_test = y_test.reshape(-1)
X = X.transpose(3, 2, 0, 1)
X_test = X_test.transpose(3, 2, 0, 1)

# Concatenate train and test
X = np.concatenate((X, X_test)).astype(np.uint8)
y = np.concatenate((y, y_test)).astype(np.uint8)

y[y == 10] = 0

# Save to parquet
import pandas as pd
df = pd.DataFrame(X.reshape(X.shape[0], -1), columns=[str(i) for i in range(X.shape[1] * X.shape[2] * X.shape[3])], dtype=np.uint8)
df["label"] = y

df.to_parquet("svhn.parquet")


