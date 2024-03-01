import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20.0, 10.0)


#Reading the Data, It was downloaded from here http://insideairbnb.com/get-the-data/
df = pd.read_csv("listings.csv")
print(df.shape)
#print(df.columns)

df = df[["accommodates","source","name","price"]]
# There are 75 columns in this dataset so I am just selecting "price" as y and 'accommodates' as x.
df.price = df.price.str.replace("$", "")
df.price = df.price.str.replace(",", "")
#print(df)
df = df.dropna(axis=0)

df["price"] = df.price.astype(float)

df['accommodates'] = df.accommodates.fillna(0)

df.price.fillna(df.price.mean() , inplace=True)

x = df['accommodates'].values
y = df['price'].values
#print(x,y)
mean_x = np.mean(x)
mean_y = np.mean(y)

n = len(x)

number = 0
denom = 0
#print(mean_x,mean_y)
for i in range(n):
    number += (x[i] - mean_x) * (y[i] - mean_y)
    denom += (x[i] - mean_x) ** 2

b1 = number / denom 
b0 = mean_y - (b1 * mean_x)

print(b1, b0)

max_x = np.max(x)  + 10
min_x = np.min(x) - 10
X = np.linspace(min_x, max_x, 50)
Y = b1 * X + b0

plt.plot(X, Y, color = "#f54d44", label = "Regression Line")
plt.scatter(x, y, c = "#22ee44", label = "Scatter Plot")

plt.xlabel("Accommodates nos")
plt.ylabel("Price in Dollars")

plt.legend()
plt.show()

ss_t = 0
ss_r = 0
for i in range(n):
    y_pred = b0 + b1 * x[i]
    ss_t += (y[i] - mean_y) ** 2
    ss_r += (y[i] - y_pred) ** 2

r2 = 1 - (ss_r/ss_t)

print(r2)