import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20.0, 10.0)


#Reading the Data, It was downloaded from here http://insideairbnb.com/get-the-data/
df = pd.read_csv("listings.csv")
print(df.shape)
#print(df.columns)

df = df[["id","source","name","price"]]
# There are 75 columns in this dataset so I will select "price" as y and 'id' as x
df.price = df.price.str.replace("$", "")
df.price = df.price.str.replace(",", "")

df = df.dropna(axis=0)

df["price"] = df.price.astype(float)

df.price.fillna(df.price.mean() , inplace=True)

x = df['id'].values
y = df['price'].values

mean_x = np.mean(x)
mean_y = np.mean(y)

n = len(x)

number = 0
denom = 0

for i in range(n):
    number += (x[i] - mean_x) * (y[i] - mean_y)
    denom += (x[i] - mean_x) ** 2

b1 = number / denom 
b0 = mean_y - (b1 * mean_x)

print(b1, b0)