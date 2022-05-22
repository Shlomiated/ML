import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from gradient_descent_utils import compute_cost, compute_gradient_descent, normalization
# take csv file and show data frame

header_list = ["Size", "Bedrooms", "Price"]
df = pd.read_csv(r"C:\Users\ShlomiAtedgi\Desktop\ML\ex1data2.txt", sep=",", names=header_list)
# print(df)

# linear regression

x = df.iloc[:, :-1].to_numpy()
x = normalization(x)
y = df.iloc[:, -1].to_numpy()
y = normalization(y)

#df.plot(x="Population",y="Profit", kind="scatter")
#plt.show()

# print(feature_metrix)

# array

iterations = 1500
alpha = 0.0001

ones_column = np.ones([len(y), 1], dtype="int")
feature_metrix = np.append(ones_column, x, axis=1)
theta_init = np.zeros([feature_metrix.shape[1], 1])
theta_f, j_history = compute_gradient_descent(theta_init, feature_metrix, y, alpha, iterations)

hf = feature_metrix.dot(theta_f).flatten()
ones, size, bedrooms = feature_metrix.T

ax = plt.axes(projection='3d')
ax.scatter3D(size, bedrooms, y, "blue")
ax.scatter3D(size, bedrooms, hf, "red")
plt.show()
print("hey")

