import modin.pandas as pd
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option("display.max_columns", 10000)
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_colwidth", 10000)
# %%
# Load the data
data = pd.read_csv("train.csv")


# 移除高缺失率的列
def remove_high_na_columns(data, threshold=0.3):
    na_rate = data.isna().mean()
    high_na_columns = na_rate[na_rate > threshold]
    logger.info(f"Removing columns with high missing values:\n{high_na_columns}")
    data = data.drop(columns=high_na_columns.index)
    return data


# 移除重复率高的列
def remove_one_value_columns(data: pd.DataFrame, threshold=0.9):
    duplicate_rate = data.apply(lambda col: col.nunique())
    high_duplicate_columns = duplicate_rate[duplicate_rate == 1]
    logger.info(
        f"Removing columns with high duplicate values:\n{high_duplicate_columns}"
    )
    data = data.drop(columns=high_duplicate_columns.index)
    return data


# 关于 manufacturd 列，和 reg_date 列，其中manufactured 是汽车的制造年份，reg_date 是汽车的注册日期。 我们可以计算发现绝大部分的 manufactured 和 reg_date是同年的，因此我们考虑用reg_date填充manufactured列的缺失值。然后移除reg_date列。
def process_manufactured(data):
    data["reg_date"] = pd.to_datetime(data["reg_date"], format=r"%d-%b-%Y")
    data["reg_year"] = data["reg_date"].dt.year

    # same_year_count = (abs(data["manufactured"] - data["reg_year"]) <= 1).sum()
    # diff_greater_than_2_years_count = (
    #     abs(data["manufactured"] - data["reg_year"]) >= 2
    # ).sum()
    # # print(f"Same year count: {same_year_count}")
    # # print(f"Difference greater than 2 years count: {diff_greater_than_2_years_count}")

    data["manufactured"] = data["manufactured"].fillna(data["reg_year"] - 2).astype(int)
    data["manufactured"] = data["manufactured"].astype("int")
    data = data.drop(columns=["reg_date", "reg_year"])
    logger.info("Filled missing values in manufactured, removed reg_date column")

    return data


# 接下来考虑fill 掉make列的缺失值，我们认为 相同的model, 它的制造商很有可能是一样的，因此可以使用model列的make来填充。 因此我们先检查model列的make是否唯一。如果对于一个model来说，make是唯一的，我们就可以这么填充。


def fill_make(data):
    data["make"] = data["make"].fillna("Unknown")

    def fill_unknown_make(group):
        most_common_make = group[group != "Unknown"].mode()
        if not most_common_make.empty:
            group[group == "Unknown"] = most_common_make[0]
        return group

    data["make"] = data.groupby("model")["make"].transform(fill_unknown_make)
    logger.info("Filled missing values in make column")

    return data


def fill_string_columns(data):
    string_cols = data.select_dtypes(include=["object"]).columns
    for col in string_cols:
        if data[col].isna().sum() > 0:
            logger.info(f"Filling missing string values in {col} column")
            data[col] = data[col].fillna("")
    return data


# %%
# 填补 mileage 列的缺失值
# 使用相同 (make, model) 组合的, 价格最接近的一个值填充。如果没有相同的 (make, model) 组合，使用 make 相同的，价格最接近的一个值填充。如果没有 make 相同的，使用价格最接近的一个值填充。，填充的时候 log 输出一下最近的价格 以及 填充的 mileage，看看是否合理.


def fill_mileage(data, k=1):
    for idx in data[data["mileage"].isnull()].index:
        pass
    return data


# %%
# 填充浮点数列的缺失值
# 其他float类型的缺失值，我们暂时使用以下方法填充缺失值，
# 使用同 make model 类别的 price 最接近的k个值的均值填充，如果同 make model 类别的数据量小于k，则使用同 make model 类别的均值填充。
# 剩余的缺失值用价格最接近的k个值的均值填充


def fill_float_columns(data, k=5):
    # 使用 IterativeImputer 进行多重插补
    mice_imputer = IterativeImputer(random_state=42, estimator=RandomForestRegressor())
    X_imputed = mice_imputer.fit_transform(data.select_dtypes(include=[np.number]))
    data_imputed = pd.DataFrame(
        X_imputed, columns=data.select_dtypes(include=[np.number]).columns
    )
    data = data_imputed
    return data


def load_data(file_path):
    data = pd.read_csv(file_path)
    data = remove_high_na_columns(data)
    data = remove_one_value_columns(data)
    data = process_manufactured(data)
    data = fill_make(data)
    data = fill_string_columns(data)
    # data = fill_mileage(data)
    # data = fill_float_columns(data)
    data.info()
    return data


if __name__ == "__main__":
    data = load_data("train.csv")
    # print(data.isna().sum())
