import numpy as np
from sklearn.preprocessing import LabelEncoder

# Categorical data
data = np.array(["Low", "Medium", "High", "Medium", "Low"])

# Create encoder
encoder = LabelEncoder()

# Fit and transform
encoded_data = encoder.fit_transform(data)

print("Original Data:", data)
print("Encoded Data:", encoded_data)