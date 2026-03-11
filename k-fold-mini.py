import numpy as np
from sklearn.model_selection import KFold

# Sample dataset
X = np.array([1,2,3,4,5,6,7,8,9,10])

# KFold object
kf = KFold(n_splits=5)

# Splitting data
for train_index, test_index in kf.split(X):
    print("Train Index:", train_index, "Test Index:", test_index)