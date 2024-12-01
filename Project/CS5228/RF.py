#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocess import load_data
from preprocess import fill_float_columns, fill_make, process_manufactured

data = load_data("train.csv")
data = data.drop(columns=["listing_id"])

# %%
# 训练一个随机森林模型
from sklearn.ensemble import RandomForestRegressor

X = data[data.select_dtypes(include=[np.number]).columns].drop("price", axis=1)
y = data["price"]

model = RandomForestRegressor(n_estimators=400, random_state=42, verbose=1, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"RMSE: {mse**0.5}")



# %%
# test data

test_data = pd.read_csv("test.csv")
test_data = test_data.drop(columns=["listing_id", 'indicative_price'])
test_data = fill_make(test_data)
test_data = process_manufactured(test_data)
test_data = test_data[test_data.select_dtypes(include=[np.number]).columns]
test_data = fill_float_columns(test_data)
test_data.info()

#%%

# 预测
y_pred = model.predict(test_data)
output = pd.DataFrame({"Id": range(len(y_pred)), "Predicted": y_pred})

#%%

output.to_csv("submission.csv", index=False)

# %%