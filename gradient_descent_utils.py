import numpy as np

def normalization(x):
    mean_x = np.mean(x,axis=0)
    std_x = np.std(x,axis=0)
    norm_x = (x - mean_x)/std_x
    
    return norm_x

def compute_cost(theta, x, y):
    h = x.dot(theta)
    dist = (h.T - y)**2
    sum_dist = np.sum(dist)
    m = len(y)
    j2 = 1/(2*m)*sum_dist

    return j2

def compute_gradient_descent(theta, x, y, a, iters):
    m = len(y)
    J_history = np.zeros(iters)
    for iter in range(iters):
        h = x.dot(theta)
        dist = h.T - y
        
        if iter == 65:
            pass
        for i in range(len(theta)):
            theta_i_temp = dist * x[:, i]
            theta_i_temp = np.sum(theta_i_temp)
            theta_i_temp = theta_i_temp * a/m
            theta[i, 0] = theta[i, 0] - theta_i_temp

        J_history[iter] = compute_cost(theta, x, y)
        if np.isnan(theta).any():
            pass
    return theta, J_history

