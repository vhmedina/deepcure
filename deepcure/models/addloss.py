from os import times_result
import tensorflow as tf
from deepcure.models.utils import mask_PE

# @tf.autograph.experimental.do_not_convert
def DeepPTMLoss_PE(event, timeset , eta_pred, lambs_coef, break_val):    
    """Loss function for DeepPTM with piecewise Exponential

    Arguments:
    - event (tf.Tensor): event indicator per individual
    - timeset (tf.Tensor): last available time point observation per individual
    - eta_pred (tf.Tensor): predictor estimation. This corresponds to log(\theta) in the PTM.
    - lambs_coef (tf.Tensor): coefficients for the piecewise exponential function.
    - break_val (tf.Tensor): break values for the piecewise exponential.

    Returns:
    - rank-1 tensor with the individual contributions to the loss function (neg log-likelihood)  

    """
    # reshape to rank-1 (vector)
    if len(event.shape) > 1:
        event = event[:,0]
    if len(timeset.shape) > 1:
        timeset = timeset[:,0]
    if len(eta_pred.shape) > 1:
        eta_pred = eta_pred[:,0]

    lambs_coef = tf.reshape(lambs_coef, [lambs_coef.shape[0]]) 

    # cast to eta dtype
    event = tf.cast(event, eta_pred.dtype)
    timeset = tf.cast(timeset, eta_pred.dtype)
    lambs_coef = tf.cast(lambs_coef, eta_pred.dtype)

    # create mask_lamb and mask_eta
    mask_lamb, mask_eta = mask_PE(timeset, break_val, eta_pred.dtype )
    
    arg_exp = tf.reduce_sum(-mask_eta * lambs_coef, axis = 1)
    lamb = tf.reduce_sum(tf.multiply(mask_lamb, lambs_coef), axis = 1) 
    l_lamb = tf.cast(tf.math.log(lamb), eta_pred.dtype) 
    arg_exp = tf.cast(arg_exp, eta_pred.dtype) 
    
    eta_shift = eta_pred + arg_exp
    r1 = l_lamb + eta_shift
    r1_event = tf.math.multiply(event, r1)
    r2 = tf.math.exp(eta_pred) - tf.math.exp(eta_shift)
    
    return r2 - r1_event


@tf.autograph.experimental.do_not_convert
def DeepPTMLoss_PE2(event, timeset , eta_pred, lambs_coef, break_val):    
    """Loss function for DeepPTM with piecewise Exponential

    Arguments:
    - event (tf.Tensor): event indicator per individual
    - timeset (tf.Tensor): last available time point observation per individual
    - eta_pred (tf.Tensor): predictor estimation. This corresponds to log(\theta) in the PTM.
    - lambs_coef (tf.Tensor): coefficients for the piecewise exponential function.
    - break_val (tf.Tensor): break values for the piecewise exponential.

    Returns:
    - rank-1 tensor with the individual contributions to the loss function (neg log-likelihood)  

    """
    # reshape to rank-1 (vector)
    if len(event.shape) > 1:
        event = event[:,0]
    if len(timeset.shape) > 1:
        timeset = timeset[:,0]
    if len(eta_pred.shape) > 1:
        eta_pred = eta_pred[:,0]

    lambs_coef = tf.reshape(lambs_coef, [lambs_coef.shape[0]]) 

    # cast to eta dtype
    event = tf.cast(event, eta_pred.dtype)
    timeset = tf.cast(timeset, eta_pred.dtype)
    lambs_coef = tf.cast(lambs_coef, eta_pred.dtype)

    # create mask_lamb and mask_eta
    mask_lamb, mask_eta = mask_PE(timeset, break_val, eta_pred.dtype )
    arg_exp = tf.reduce_sum(-tf.multiply(mask_eta, lambs_coef), axis = 1)
    lamb = tf.reduce_sum(tf.multiply(mask_lamb, lambs_coef), axis = 1) 
    l_lamb = tf.cast(tf.math.log(lamb), eta_pred.dtype) 
    arg_exp = tf.cast(arg_exp, eta_pred.dtype) 

    theta = tf.math.softplus(eta_pred)
    eta_shift = arg_exp + tf.math.log(theta)
    r1 = l_lamb + eta_shift
    r1_event = tf.math.multiply(event, r1)
    r2 = theta - tf.math.exp(eta_shift)
    
    return r2 - r1_event

# @tf.autograph.experimental.do_not_convert
def DeepPTMLoss_Weibull(event, timeset , eta_pred, weibull_coef):    
    """Loss function for DeepPTM with piecewise Exponential

    Arguments:
    - event (tf.Tensor): event indicator per individual
    - timeset (tf.Tensor): last available time point observation per individual
    - eta_pred (tf.Tensor): predictor estimation. This corresponds to log(\theta) in the PTM.
    - weibull_coef (tf.Tensor): coefficients for the Weibull function (k=concentration, lambda=scale).

    Returns:
    - rank-1 tensor with the individual contributions to the loss function (neg log-likelihood)  

    """
    # reshape to rank-1 (vector)
    if len(event.shape) > 1:
        event = event[:,0]
    if len(timeset.shape) > 1:
        timeset = timeset[:,0]
    if len(eta_pred.shape) > 1:
        eta_pred = eta_pred[:,0]
    weibull_coef = tf.reshape(weibull_coef, [weibull_coef.shape[0]]) 

    # cast to eta dtype
    event = tf.cast(event, eta_pred.dtype)
    timeset = tf.cast(timeset, eta_pred.dtype)
    weibull_coef = tf.cast(weibull_coef, eta_pred.dtype)
    k = weibull_coef[0]
    lamm = weibull_coef[1]
    
    eta_shift = eta_pred + -(timeset/lamm)**k 
    r1 = eta_shift +tf.math.log(k) - k * tf.math.log(lamm) + tf.math.scalar_mul((k-1), tf.math.log(timeset))
    r1_event = tf.math.multiply(event, r1)
    r2 = tf.math.exp(eta_pred) - tf.math.exp(eta_shift)
    
    return r2 - r1_event

@tf.autograph.experimental.do_not_convert
def DeepPTMLoss_Weibull2(event, timeset , eta_pred, weibull_coef):    
    """Loss function for DeepPTM with piecewise Exponential

    Arguments:
    - event (tf.Tensor): event indicator per individual
    - timeset (tf.Tensor): last available time point observation per individual
    - eta_pred (tf.Tensor): predictor estimation. This corresponds to log(\theta) in the PTM.
    - weibull_coef (tf.Tensor): coefficients for the Weibull function (k=concentration, lambda=scale).

    Returns:
    - rank-1 tensor with the individual contributions to the loss function (neg log-likelihood)  

    """
    # reshape to rank-1 (vector)
    if len(event.shape) > 1:
        event = event[:,0]
    if len(timeset.shape) > 1:
        timeset = timeset[:,0]
    if len(eta_pred.shape) > 1:
        eta_pred = eta_pred[:,0]
    weibull_coef = tf.reshape(weibull_coef, [weibull_coef.shape[0]]) 

    # cast to eta dtype
    event = tf.cast(event, eta_pred.dtype)
    timeset = tf.cast(timeset, eta_pred.dtype)
    weibull_coef = tf.cast(weibull_coef, eta_pred.dtype)
    k = weibull_coef[0]
    lamm = weibull_coef[1]
    

    theta = tf.math.softplus(eta_pred)
    eta_shift = -(timeset/lamm)**k + tf.math.log(theta)
    r1 = eta_shift +tf.math.log(k) - k * tf.math.log(lamm) + tf.math.scalar_mul((k-1), tf.math.log(timeset))
    r1_event = tf.math.multiply(event, r1)
    r2 = theta - tf.math.exp(eta_shift)
    
    return r2 - r1_event