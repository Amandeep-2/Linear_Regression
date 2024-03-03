import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("listings.csv")
print(df.columns)

df = df[["host_listings_count","accommodates","beds","price"]]

df["beds"] = df.beds.fillna(int(df.beds.median()))

df["accommodates"] = df.accommodates.fillna(int(df.accommodates.median()))

df["host_listings_count"] = df.host_listings_count.fillna(int(df.host_listings_count.median()))

df.price = df.price.str.replace("$", "")
df.price = df.price.str.replace(",", "")

df["price"] = df.price.astype(float)

df["price"] = df.price.fillna(df.price.median())

print(df.describe)
reg = linear_model.LinearRegression()
reg.fit(df[['host_listings_count','accommodates','beds']],df.price)

print(reg.coef_, reg.intercept_)

print(reg.predict([[5,5,5]]))