import numpy as np

# Vector
vector = np.array([1, 2, 3, 4])
print("Vector:", vector)

# Matrix
matrix = np.array([[1, 2],
                   [3, 4]])

print("Matrix:")
print(matrix)

# Vector addition
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
print("Vector Addition:", v1 + v2)

# Matrix addition
m1 = np.array([[1,2],[3,4]])
m2 = np.array([[5,6],[7,8]])
print("Matrix Addition:")
print(m1 + m2)

# Matrix multiplication
print("Matrix Multiplication:")
print(np.dot(m1, m2))