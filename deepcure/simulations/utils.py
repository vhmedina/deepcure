import numpy as np
import tensorflow as tf
import pandas as pd


# -------------- inverse of exponentail CDF for time simulation -------------- #
def vinv_F_exponential(lamb, n, N_i, seed):
    np.random.seed(seed)
    np.seterr(divide='ignore', invalid='ignore')
    t = -np.log(np.random.uniform(size=n))/(lamb * N_i)
    return t


 # ----------------------- Error metrics for simulation ----------------------- #
def metrics(data_real, data_est):
    # calculate mean squared difference of survival S1
    S1_real = np.exp(-data_real['event_time']) # assuming exp with rate=1
    S1_est = data_est['S1']
    S1_mse = np.mean((S1_real-S1_est)**2)
    # calculate mean squared difference of eta
    eta_mse = np.mean((data_real['eta']- data_est['eta'])**2)
    # calculate mean squared difference of Sp
    Sp_real = np.exp(-np.exp(data_real['eta']) * (1-S1_real) )
    Sp_est = np.exp(-np.exp(data_est['eta']) * (1-S1_est) )
    Sp_mse = np.mean((Sp_real-Sp_est)**2)
    return Sp_mse, eta_mse, S1_mse

# ------------------------- Orthogonalization with tf ------------------------ #
def tf_ortho(U,X):
    X = tf.convert_to_tensor(X)
    U = tf.convert_to_tensor(U)
    Q = tf.linalg.qr(X, full_matrices=False).q
    QQT = tf.linalg.matmul(Q, tf.linalg.matrix_transpose(Q))
    res = tf.subtract(U, tf.linalg.matmul(QQT, U))
    return res.numpy()