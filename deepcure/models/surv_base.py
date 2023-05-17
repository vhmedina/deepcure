from tensorflow import keras

class sbase(keras.Model):
    """Generic class for survival models.
    """

    def surv_predict(self, inputs, type=None):
        """Survival prediction for `input`.
        
        Arguments:
            input {tf.tensor or np.array} -- Input

        Returns:
            [tensor or np.ndarray] -- Survival predictions

        """
        raise NotImplementedError

    def haz_predict(self, inputs):
        """Survival prediction for `input`.
        
        Arguments:
            input {tf.tensor or np.array} -- Input

        Returns:
            [tensor or np.ndarray] -- Survival predictions
        """
        raise NotImplementedError
    


