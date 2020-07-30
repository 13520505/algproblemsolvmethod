import numpy as np
import pandas as pd

FEATURE_PATH = "./resources/Features data set.csv"

def linear_regression(data,learning_rate=0.01,num_iters=300):
    # 0 -> n-1: feature, n: y value
    numFeature = data.shape[1]
    X, y = data[:,:numFeature-1], data[:, numFeature-1]
    n = y.size
    # Norm data
    X_norm,mu,sigma = feature_norm(X)
    # Add intercept term to X
    X = np.concatenate([np.ones((n, 1)), X_norm], axis=1)
    # Cal grad
    # Choose some alpha value - change this
    alpha = learning_rate
    num_iters = num_iters

    # init theta and run gradient descent
    theta = np.zeros(3)
    theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
    return theta, J_history

def compute_cost(X, y, theta):
    m = y.shape[0]
    J = 0
    J = np.sum((X @ theta - y)**2 )/ 2 /m
    return J

def feature_norm(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    # norm = (X - mean) / std
    mu = np.mean(X,axis=0)
    sigma2 = np.std(X,0,ddof=1)
    sigma = np.std(X,0)
    print(sigma)
    print(sigma2)
    print("----")
    X_norm = (X - mu)/sigma
    # store mu, sigma for new pred
    return X_norm,mu,sigma

def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(0,num_iters):
        er = (X @ theta - y) @ X
        theta -= (alpha / m) * er
        print(theta)
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history


def preprocess_data(path):
    dataset = pd.read_csv(path)
    store1 = get_store_feature(dataset,1)
    print(store1)
    return None

def get_store_feature(ds,store_id):
    ds = ds[ds["Store"] == store_id]
    return ds

# preprocess_data(FEATURE_PATH)

ds = np.loadtxt("test.txt",delimiter=',')
theta, J = linear_regression(ds)
print(theta)