import numpy as np
import pandas as pd


def load_data(base_path):
    train_path = base_path+"train.csv"
    features_path = base_path+"features.csv"
    stores_path = base_path+"stores.csv"
    dataset = pd.read_csv(train_path, names=['Store','Dept','Date','weeklySales','isHoliday'],sep=',', header=0)
    features = pd.read_csv(features_path,sep=',', header=0,
                        names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4',
                                'MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])
    stores = pd.read_csv(stores_path, names=['Store','Type','Size'],sep=',', header=0)
    dataset = dataset.merge(stores, how='left').merge(features, how='left')
    return dataset


def preprocess_data(ds):
    ds = pd.get_dummies(ds, columns=["Type"])
    ds[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = ds[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)
    return ds

def get_store_feature(ds,store_id,dept_id):
    ds = ds[ds["Store"] == store_id ]
    ds = ds[ds["Dept"] == dept_id ]
    
    return ds

def rearrage(ds):
    PREDICTED_COL_NAME = ["weeklySales"]
    features_col_name = ds.columns.drop(PREDICTED_COL_NAME).tolist()
    re_arrange = features_col_name + PREDICTED_COL_NAME
    ds = ds[re_arrange]
    return ds

# preprocess_data(FEATURE_PATH)
BASE_PATH = "./resources/"
ds = load_data(BASE_PATH)
ds = preprocess_data(ds)
store1_dept1 = get_store_feature(ds,1,1)
print(store1_dept1)
re =rearrage(store1_dept1)
print(re)