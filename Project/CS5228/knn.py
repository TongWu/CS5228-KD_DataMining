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
pd.set_option("display.max_columns", 10000)
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_colwidth", 10000)

data = pd.read_csv("train.csv")

# 初步的方法
# 1. 使用knn回归模型，使用make和model作为分类特征，其他特征作为数值特征 进行预测

categorical_features = ['make', 'model']
numerical_features = [col for col in data.columns if col not in categorical_features + ['price']]

data[categorical_features] = data[categorical_features].astype(str)
data[numerical_features] = data[numerical_features].apply(pd.to_numeric)

# 定义目标变量和特征
target = 'price'
X = data.drop(columns=[target])
y = data[target]

# 划分数据集，80%训练集，20%测试集，随机种子为42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 重置索引以确保一致性
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# 标准化数值特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# 将缩放后的数值特征转换回 DataFrame 并与分类特征合并
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numerical_features)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numerical_features)

X_train_final = pd.concat([X_train_scaled_df, X_train[categorical_features]], axis=1)
X_test_final = pd.concat([X_test_scaled_df, X_test[categorical_features]], axis=1)



# 初始化预测结果数组
y_pred = np.empty(len(X_test_final))
y_pred[:] = np.nan

# 按照 ('make', 'model') 分组预测
group_make_model = X_test_final.groupby(['make', 'model'])
# 定义要尝试的 k 值范围
k_values = range(1, 11)  # 例如，从 1 到 10
rmse_values = []

# 遍历每个 k 值
for k in k_values:
    print(f"Processing k={k}")
    # 初始化预测结果数组
    y_pred = np.empty(len(X_test_final))
    y_pred[:] = np.nan
    
    # 遍历每个组进行预测
    for (make, model), group_indices in group_make_model.groups.items():
        # 1. 找到训练集中与当前组相同 make 和 model 的样本
        mask = (X_train['make'] == make) & (X_train['model'] == model)
        if mask.any():
            X_subset = X_train_scaled_df[mask]
            y_subset = y_train[mask]
        else:
            # 2. 如果没有相同的 make 和 model，找相同 make 的样本
            mask_make = X_train['make'] == make
            if mask_make.any():
                X_subset = X_train_scaled_df[mask_make]
                y_subset = y_train[mask_make]
            else:
                # 3. 如果没有相同 make 的样本，则使用全局训练数据
                X_subset = X_train_scaled_df
                y_subset = y_train
        
        # 确保 k 不超过样本数量
        current_k = min(k, len(X_subset))
        if current_k == 0:
            # 如果没有可用的样本，跳过预测
            continue
        
        # 初始化 KNN 模型
        knn = KNeighborsRegressor(n_neighbors=current_k, algorithm='auto')
        # 训练 KNN 并进行预测
        knn.fit(X_subset, y_subset)
        X_test_group = X_test_scaled_df.iloc[group_indices]
        y_pred[group_indices] = knn.predict(X_test_group)
    
    # 评估模型
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append(rmse)
    print(f"RMSE for k={k}: {rmse:.2f}")

# 绘制 RMSE 与 k 的关系图
plt.figure(figsize=(10, 6))
plt.plot(k_values, rmse_values, marker='o')
plt.xlabel('k 值')
plt.ylabel('RMSE')
plt.title('KNN 回归中 k 值与 RMSE 的关系')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# 选择最佳 k 值（最小 RMSE）
best_k = k_values[np.argmin(rmse_values)]
best_rmse = min(rmse_values)
print(f"最佳 k 值为 {best_k}，对应的 RMSE 为 {best_rmse:.2f}")

# 使用最佳 k 值重新进行预测并展示误差最大的5个样本
# 初始化预测结果数组
y_pred_best = np.empty(len(X_test_final))
y_pred_best[:] = np.nan

for (make, model), group_indices in group_make_model.groups.items():
    # 1. 找到训练集中与当前组相同 make 和 model 的样本
    mask = (X_train['make'] == make) & (X_train['model'] == model)
    if mask.any():
        X_subset = X_train_scaled_df[mask]
        y_subset = y_train[mask]
    else:
        # 2. 如果没有相同的 make 和 model，找相同 make 的样本
        mask_make = X_train['make'] == make
        if mask_make.any():
            X_subset = X_train_scaled_df[mask_make]
            y_subset = y_train[mask_make]
        else:
            # 3. 如果没有相同 make 的样本，则使用全局训练数据
            X_subset = X_train_scaled_df
            y_subset = y_train
    
    # 确保 k 不超过样本数量
    current_k = min(best_k, len(X_subset))
    if current_k == 0:
        continue
    
    # 初始化 KNN 模型
    knn = KNeighborsRegressor(n_neighbors=current_k, algorithm='auto')
    # 训练 KNN 并进行预测
    knn.fit(X_subset, y_subset)
    X_test_group = X_test_scaled_df.iloc[group_indices]
    y_pred_best[group_indices] = knn.predict(X_test_group)

# 计算绝对误差并创建结果 DataFrame
errors = np.abs(y_test - y_pred_best)
results_df = X_test.copy()
results_df['Actual Price'] = y_test
results_df['Predicted Price'] = y_pred_best
results_df['Absolute Error'] = errors

# 找出误差最大的5个样本
top_errors = results_df.nlargest(5, 'Absolute Error')

print(f"\n误差最大的5个样本（使用最佳 k={best_k}）：")
print(top_errors)