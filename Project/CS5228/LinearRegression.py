# %%
import modin.pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
import multiprocessing
from tqdm_joblib import tqdm_joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

pd.set_option("display.max_columns", 10000)
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_colwidth", 10000)
# %%
# Load the data
from preprocess import load_data

data = load_data("train.csv")
data = data.dropna()
data.info()

# %%
# 取出所有的数值型特征
numerical_features = data.select_dtypes(include=[np.number]).columns
data = data[numerical_features]
data = data.drop(columns=["listing_id"])
data.info()

# %%
# 计算每个特征的相关性
correlation = data.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Matrix")
plt.show()

# %%
# 主成分分析
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
pca = PCA(n_components=5)
data_pca = pca.fit_transform(data_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:")
print(explained_variance_ratio)

#%%
# 取前4个主成分， 作为新的特征， 在这上面进行线性回归
data_pca = data_pca[:, :4]
X_train, X_test, y_train, y_test = train_test_split(data_pca, data["price"], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse**0.5}")



