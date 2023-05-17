import tensorflow as tf
import numpy as np

@tf.function
def mask_PE(timeset, break_val, eta_dtype):
    """Create matrices for Piecewise Exponential

    Arguments:
    - timeset (tf.Tensor): last available time point observation per individual.
    - break_val (tf.Tensor): break values for the piecewise exponential.
    - eta_dtype: dtype of eta_predictions (log(theta) in the PTM). 

    Return:
    - Two rank-2 tensors. mask_lamb which indicates the corresponding lambda coefficient and mask_eta which has the corresponding time intervals
    """
    break_val = tf.cast(break_val, eta_dtype)
    break_val1 = break_val[1:break_val.shape[0]]
    break_val0 = break_val[0:(break_val.shape[0]-1)]

    mask_aux = tf.math.greater(tf.reshape(timeset,(-1,1)), break_val1)
    mask_lamb = tf.one_hot(tf.reduce_sum(tf.where(mask_aux, 1, 0), axis=1), break_val1.shape[0], dtype=eta_dtype)

    delta_interv1 = break_val[1:break_val.shape[0]]-break_val0[0:(break_val.shape[0]-1)]
    delta_interv1 = tf.cast(delta_interv1, dtype = eta_dtype)
    delta_interv2 = tf.reshape(timeset,(-1,1))-break_val0
    delta_interv2 = tf.cast(delta_interv2, dtype = eta_dtype)

    mask_eta = tf.multiply(tf.cast(tf.where(mask_aux, 1., 0.), dtype=eta_dtype), delta_interv1) + tf.multiply(mask_lamb, delta_interv2)

    mask_eta = tf.cast(mask_eta, eta_dtype)

    return mask_lamb, mask_eta


def piecewise_exp_surv(time_event, break_val, lambs_coef):
    """Piecewise Exponential Survival Function

    Arguments:
    - time_event (tf.Tensor): last available time point observation per individual.
    - break_val (tf.Tensor): break values for the piecewise exponential.
    - lambs_coef (tf.Tensor): lambda coefficients for the piecewise exponential.

    Return:
    - tf.Tensor: Piecewise Exponential Survival Function
    """
    # create mask_lamb and mask_eta
    _, mask_eta = mask_PE(time_event, break_val, time_event.dtype )
    arg_exp = -tf.reduce_sum(mask_eta * lambs_coef, axis = 1)
    S = tf.reshape(tf.math.exp(arg_exp),(-1,1)) # reshape to rank-2 tensor
    return S

def piecewise_exp_pdf(time_event, break_val, lambs_coef):
    """Piecewise Exponential Probability Density Function

    Arguments:
    - time_event (tf.Tensor): last available time point observation per individual.
    - break_val (tf.Tensor): break values for the piecewise exponential.
    - lambs_coef (tf.Tensor): lambda coefficients for the piecewise exponential.

    Return:
    - tf.Tensor: Piecewise Exponential Probability Density Function
    """
    # create mask_lamb and mask_eta
    mask_lamb, mask_eta = mask_PE(time_event, break_val, time_event.dtype )
    lamb = tf.reduce_sum(tf.multiply(mask_lamb, lambs_coef), axis = 1) 
    arg_exp = -tf.reduce_sum(mask_eta * lambs_coef, axis = 1)
    f = tf.reshape(tf.multiply(lamb,tf.exp(arg_exp)),(-1,1)) # reshape to rank-2 tensor
    return f


def weibull_surv(time_event, weibull_coef):
    """Weibull Survival Function

    Arguments:
    - time_event (tf.Tensor): last available time point observation per individual.
    - weibull_coef (tf.Tensor): weibull coefficients for the weibull distribution.

    Return:
    - tf.Tensor: Weibull Survival Function
    """
    weibull_coef = tf.reshape(weibull_coef, [weibull_coef.shape[0]]) 
    k = weibull_coef[0] 
    lamm = weibull_coef[1]
    S =  tf.reshape(tf.math.exp(-tf.math.pow(time_event/lamm, k)),(-1,1)) # reshape to rank-2 tensor
    return S
    
def weibull_pdf(time_event, weibull_coef):
    """Weibull Probability Density Function

    Arguments:
    - time_event (tf.Tensor): last available time point observation per individual.
    - weibull_coef (tf.Tensor): weibull coefficients for the weibull distribution.

    Return:
    - tf.Tensor: Weibull Probability Density Function
    """
    weibull_coef = tf.reshape(weibull_coef, [weibull_coef.shape[0]]) 
    k = weibull_coef[0] 
    lamm = weibull_coef[1]
    f =  tf.reshape((k/lamm)*tf.math.pow(time_event/lamm, (k-1)) * tf.math.exp(-tf.math.pow(time_event/lamm, k)),(-1,1)) # reshape to rank-2 tensor
    return f

@tf.function
def ortho(U, X):
    """Orthogonalize a matrix U with respect to a matrix X
    
    Arguments:
    - U (tf.Tensor): matrix to be orthogonalized
    - X (tf.Tensor): matrix with respect to which U is orthogonalized

    Return:
    - tf.Tensor: orthogonalized matrix U

    
    """
    
    # create the projection matrix
    Q = tf.linalg.qr(X, full_matrices=False).q
    QQT = tf.linalg.matmul(Q, tf.linalg.matrix_transpose(Q))
    return (tf.subtract(U, tf.linalg.matmul(QQT, U)))