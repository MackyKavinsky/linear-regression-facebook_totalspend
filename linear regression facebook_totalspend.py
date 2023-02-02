import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv("facebook_data.csv")

# Preprocess the data
df = df.dropna()
df = (df - df.mean()) / df.std()

# Split the data
X = df.drop("total_spend", axis=1)
y = df["total_spend"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
Note that this is a simple example and you may need to make additional modifications to the code depending on your specific use case.




