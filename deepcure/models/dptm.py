import tensorflow as tf
import numpy as np
from deepcure import models
from deepcure import performance as perf

class _EndpointDeepPTM_PE(tf.keras.layers.Layer):
    """Endpoint layer for DeepPTM with Piecewise Exponential function 

    Arguments:
    - break_val (tf.Tensor): break values for the piecewise exponential.
    - parametrization (int): parametrization of the loss function.

    Returns:
    - rank-1 tensor with the individual \eta prediction. When training, it adds the \lambda loss contribution. 
    """
    def __init__(self, break_val, parametrization=None, name=None):
        super(_EndpointDeepPTM_PE, self).__init__(name=name)
        if parametrization == 1:
            self.loss_fn = models.addloss.DeepPTMLoss_PE
        elif parametrization == 2:
            self.loss_fn = models.addloss.DeepPTMLoss_PE2
        else :
            raise ValueError('Unknown parametrization')
        self.n_t_breaks = len(break_val)-1
        self.break_val = break_val
        self.parametrization = parametrization


    def build(self, input_shape):

        self._lamms=self.add_weight(name = 'lambdas',
                                shape = (self.n_t_breaks,),
                                initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.),
                                trainable=True,
                                constraint = tf.keras.constraints.NonNeg() )

        super(_EndpointDeepPTM_PE, self).build(input_shape)

    def call(self, eta_pred, event=None,timeset=None):
        if event is not None:
            # Compute the loss value and add it to the layer.
            loss = self.loss_fn(event, timeset, eta_pred, self._lamms, self.break_val)
            self.add_loss(loss)

        # Return eta
        return eta_pred
    
    def get_config(self):
        config = super(_EndpointDeepPTM_PE, self).get_config() 
        config.update({
        "loss_fn": self.loss_fn,
        "n_t_breaks": self.n_t_breaks,
        "break_val": self.break_val.numpy(),
        "parametrization": self.parametrization,})
        return config

class _EndpointDeepPTM_Weibull(tf.keras.layers.Layer):
    """Endpoint layer for DeepPTM with Weibull function 

    Arguments:
    - parametrization (int): parametrization of the loss function.

    Returns:
    - rank-1 tensor with the individual \eta prediction. When training, it adds the \weibull_coef loss contribution. 
    """
    def __init__(self, parametrization=None, name=None):
        super(_EndpointDeepPTM_Weibull, self).__init__(name=name)
        if parametrization == 1:
            self.loss_fn = models.addloss.DeepPTMLoss_Weibull 
        elif parametrization == 2:
            self.loss_fn = models.addloss.DeepPTMLoss_Weibull2 
        else :
            raise ValueError('Unknown parametrization')
        self.parametrization = parametrization

    def build(self, input_shape):

        self._lamms=self.add_weight(name = 'weibull_coef',
                                shape = (2,),
                                initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.),
                                trainable=True,
                                constraint = tf.keras.constraints.NonNeg()
                                 )

        super(_EndpointDeepPTM_Weibull, self).build(input_shape)

    def call(self, eta_pred, event=None,timeset=None):
        if event is not None:
            # Compute the loss value and add it to the layer.
            loss = self.loss_fn(event, timeset, eta_pred, self._lamms)
            self.add_loss(loss)

        # Return eta
        return eta_pred
    
    def get_config(self):
        config = super(_EndpointDeepPTM_Weibull, self).get_config() 
        config.update({
        "loss_fn": self.loss_fn,
        "parametrization": self.parametrization,})
        return config
    
class _AddCol_layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(_AddCol_layer, self).__init__(**kwargs)
    
    def call(self, inputs):
        ones = tf.ones_like(tf.gather(inputs, [0], axis=1))
        return tf.concat([ones, inputs], axis=1)
    
    def get_config(self):
        config = super(_AddCol_layer, self).get_config()
        return config
    
class _Ortho_layer(tf.keras.layers.Layer):
    """Layer to orthogonalize the U matrix

    Arguments:
    - deact_test (bool): if True, the layer is deactivated during testing.

    Returns:
    - rank-2 tensor with the orthogonalized U matrix.
    
    """
    def __init__(self, deact_test):
        self.deact_test = deact_test
        self.add_intercept = _AddCol_layer()
        super(_Ortho_layer, self).__init__()
        
    def call(self, U, X, training = None):
        if not self.deact_test or training:
            U = tf.cast(U, tf.float64)
            X = self.add_intercept(X)
            X = tf.cast(X, tf.float64)
            Q = tf.linalg.qr(X, full_matrices=False).q
            QQT = tf.linalg.matmul(Q, tf.linalg.matrix_transpose(Q))
            res = tf.subtract(U, tf.linalg.matmul(QQT, U))
            return res
        else:
            return U
        
class _Extract_cols(tf.keras.layers.Layer):
    """Layer to extract columns from a rank-2 tensor
    
    Arguments:
    - cols (list of int): list of columns to extract.

    Returns:
    - rank-2 tensor with the selected columns.

    """
    def __init__(self, cols, **kwargs):
        super(_Extract_cols, self).__init__(**kwargs)
        self.cols = cols
        self.supports_masking = True

    def call(self, inputs):
        return tf.gather(inputs, self.cols, axis=1)

    def get_config(self):
        config = super(_Extract_cols, self).get_config()
        config.update({'cols': self.cols})
        return config

class DeepPTM(models.surv_base.sbase):
    """The Promotion Time Cure Model with NNet for \eta and Piecewise Exponential for F(t)

    Arguments:
    - stack_eta (list of keras.layers): list of layer representing the \eta architecture.
    - break_val (tf.Tensor): break values for the piecewise exponential.
    - t_func (str): name of the function to use for the F(t).
    - parametrization (int): parametrization of the loss function. Default is 1 corresponding to \theta = exp(\eta).\\
        2 corresponds to \theta = softplus(\eta).

    Returns:
    - keras.Model for the PTM.
    """
    def __init__(self, stack_eta, t_func, break_val=None, parametrization=1, name=None, cols_ortho=None, deact_test=False):
        super(DeepPTM, self).__init__(name=name)
        self.stack_eta = stack_eta
        self.break_val = break_val
        self.t_func = t_func
        self.parametrization = parametrization
        self.cols_ortho = cols_ortho
        self.deact_test = deact_test
        self.len_stack = len(stack_eta)
        if cols_ortho!=None:
            # check if cols_ortho is a list of column names
            if type(cols_ortho)==list or type(cols_ortho)==np.ndarray or cols_ortho=='all':
                self.ortholayer = _Ortho_layer(deact_test=deact_test)
                if cols_ortho!='all':
                    self.extract = _Extract_cols(cols_ortho)
                self.linear = tf.keras.layers.Dense(1, activation='linear', use_bias=True, name='out_lin_layer',dtype=tf.float64)
                self.addlayer = tf.keras.layers.Add(dtype=tf.float64)
            else:
                raise ValueError('cols_ortho must be either None, "all" or a list of column names')
        if t_func=='weibull':
            self.endpoint = _EndpointDeepPTM_Weibull(parametrization, name= "endpoint")
        elif t_func=='pe':
            self.endpoint = _EndpointDeepPTM_PE(break_val, parametrization, name= "endpoint")
        else:
            raise ValueError('t_func must be either weibull or pe')

    def call(self, inputs):
        # define inputs
        x_inputs = inputs['x']
        label_event = inputs['label_event']
        time_event = inputs['time_event']

        # create \eta
        for i,layer in enumerate(self.stack_eta):
            if (i == self.len_stack-1) and self.cols_ortho!=None:
                if self.cols_ortho!='all':
                    x_inputs_lin = self.extract(inputs['x'])
                else:
                    x_inputs_lin = inputs['x']
                x_inputs = self.ortholayer(x_inputs, x_inputs_lin, training=True)
                x_inputs = layer(x_inputs)
                lin_eff = self.linear(x_inputs_lin)
                x_inputs = self.addlayer([x_inputs, lin_eff])                    
            else:
                x_inputs = layer(x_inputs)
        eta_plus = self.endpoint(eta_pred = x_inputs, event = label_event, timeset = time_event)
        return eta_plus

    def get_config(self):
        config = super(DeepPTM, self).get_config()
        config.update({'stack_eta': self.stack_eta,
                    'break_val': self.break_val,
                    't_func': self.t_func,
                    'parametrization': self.parametrization,
                    'endpoint': self.endpoint,
                    'cols_ortho': self.cols_ortho,
                    'deact_test': self.deact_test,
                    'len_stack': self.len_stack,})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def get_weights2(self):
        lay_weights = []
        for lay in self.layers:
            for w in lay.weights:
                lay_weights.append(w.numpy())
        return lay_weights
    
    def eta_predict(self, inputs):  
        x_inputs = inputs['x']
        shape_x = x_inputs.shape
        if len(shape_x)==1:
            shape_x = tf.reshape(x_inputs,(-1,1)).shape 
        x = tf.keras.Input((shape_x[1],))
        for i,layer in enumerate(self.stack_eta):
            if i==0:
                xi=layer(x)
            if (not isinstance(layer, tf.keras.layers.Dropout)) and i>0:
                if (i == self.len_stack-1) and self.cols_ortho!=None:
                    if self.cols_ortho!='all':
                        x_lin = self.extract(x)
                    else:
                        x_lin = x  
                    xi = self.ortholayer(xi, x_lin, training=False)
                    xi = layer(xi)
                    lin_eff = self.linear(x_lin)
                    xi = self.addlayer([xi, lin_eff])
                else:
                    xi = layer(xi)
        eta_plus = self.endpoint(eta_pred = xi, event = None, timeset = None)
        inference_model = tf.keras.Model([x], eta_plus)
        inference_model.set_weights(self.get_weights2())
        pred = inference_model.predict(x_inputs, verbose=0)
        return pred

    def theta_predict(self, inputs):
        if self.parametrization == 1:
            eta = self.eta_predict(inputs)
            theta = tf.exp(eta)
        elif self.parametrization == 2:
            eta = self.eta_predict(inputs)
            theta = tf.nn.softplus(eta)
        return theta

    def surv_predict(self, inputs, type='S1'):
        # define inputs
        time_event = inputs['time_event']
        if self.t_func == 'weibull':
            # retrieve the weibull coef
            weibull_coef = tf.cast(self.get_layer('endpoint').get_weights()[0], time_event.dtype)
            # compute the survival function
            S_pred = models.utils.weibull_surv(time_event, weibull_coef)
        elif self.t_func == 'pe':
            # retrieve the break values
            break_val = tf.cast(self.break_val, time_event.dtype)
            # retrieve lambda coefs
            lamms = tf.reshape(tf.cast(self.get_layer('endpoint').get_weights()[0], time_event.dtype), (1,-1))
            # compute the survival function
            S_pred = models.utils.piecewise_exp_surv(time_event, break_val, lamms)
        if type=='Sp':
            theta = tf.cast(self.theta_predict(inputs), time_event.dtype)
            S_pred = tf.exp(-theta*(1-S_pred))
        return S_pred.numpy()
    
    def _surv_predict_mat(self, inputs, time_eval=None):
        n_ids = inputs['time_event'].shape[0]
        if time_eval is None: # evaluate in the time_eval
            n_times = n_ids
            surv_mat = np.empty((n_ids, n_times), dtype=np.float32)
            for j in range(n_times):
                inputs_aux = inputs.copy()
                inputs_aux['time_event'] = tf.repeat(inputs['time_event'][j], n_ids, axis=0)
                surv_mat[:,j] = self.surv_predict(inputs_aux, type='Sp')[:,0].astype(np.float32)
        else:
            n_times = time_eval.shape[0]
            surv_mat = np.empty((n_ids, n_times), dtype=np.float32)
            for j in range(n_times):
                inputs_aux = inputs.copy()
                inputs_aux['time_event'] = tf.repeat(time_eval[j], n_ids, axis=0)
                surv_mat[:,j] = self.surv_predict(inputs_aux, type='Sp')[:,0].astype(np.float32)
        return surv_mat
    
    # survival function for input i
    def _surv_predict_i(self, inputs, index_i):
        n_ids = inputs['time_event'].shape[0]
        surv_mat_i = np.empty((n_ids, 1), dtype=np.float32)
        inputs_aux = inputs.copy()
        inputs_aux['time_event'] = tf.repeat(inputs['time_event'][index_i], n_ids, axis=0)
        surv_mat_i[:,0] = self.surv_predict(inputs_aux, type='Sp')[:,0].astype(np.float32)
        return surv_mat_i

    def haz_predict(self, inputs):
        # define inputs
        time_event = inputs['time_event']
        theta = tf.cast(self.theta_predict(inputs), time_event.dtype)
        if self.t_func == 'weibull':
            # retrieve the weibull coef
            weibull_coef = tf.cast(self.get_layer('endpoint').get_weights()[0], time_event.dtype)
            # compute the pdf function
            f_pred = models.utils.weibull_pdf(time_event, weibull_coef)
            h_pred = theta * f_pred
        elif self.t_func == 'pe':
            # retrieve the break values
            break_val = tf.cast(self.break_val, time_event.dtype)
            # retrieve lambda coefs
            lamms = tf.reshape(tf.cast(self.get_layer('endpoint').get_weights()[0], time_event.dtype), (1,-1))
            # compute the pdf function
            f_pred = models.utils.piecewise_exp_pdf(time_event, break_val, lamms)
            h_pred = theta * f_pred
        return h_pred.numpy()

    def performance(self, inputs, metric: str, time_eval=None, inputs_train=None, format='mean'):
        if metric == 'auc_c':
            prob_cured = np.exp(-self.theta_predict(inputs).numpy())
            return perf.metrics.get_auc(prob_cured)
        elif metric == 'brier':
            surv_mat = self._surv_predict_mat(inputs, time_eval)
            return perf.metrics.get_brier_score(inputs_train['time_event'].numpy(), inputs_train['label_event'].numpy(), inputs['time_event'].numpy(), inputs['label_event'].numpy(), surv_mat, time_eval)
        elif metric == 'ibs':
            surv_mat = self._surv_predict_mat(inputs, time_eval)
            perf_b = perf.metrics.get_brier_score(inputs_train['time_event'].numpy(), inputs_train['label_event'].numpy(), inputs['time_event'].numpy(), inputs['label_event'].numpy(), surv_mat, time_eval)
            return perf.metrics.get_ibs(perf_b)




        