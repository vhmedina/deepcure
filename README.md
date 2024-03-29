# `deepcure` package

`deepcure` is a python package for survival cure models based on [TensorFlow](https://www.tensorflow.org/)/[Keras](https://keras.io/).

## Installation

This package is in its early development stage.

To install the development version from GitHub:
```bash
pip install git+https://github.com/vhmedina/deepcure.git
```

## Usage

```python
from deepcure.models import DeepPTM
from tensorflow.keras.layer import Dense

stack_eta = [Dense(1, activation='linear', name = "eta")] # Definition of the stack of layers for eta in promotion time cure model
model = DeepPTM(stack_eta = stack_eta, t_func='pe', break_val=break_val) 
model.compile(optimizer='adam')
model.fit(x_train, y_train, epochs=10)
```

See the [tutorials](../main/tutorials) for more details and features.

## How to cite `deepcure`?
For the methodology, please cite the following

```
@article{medina2023deep,
  title={The Deep Promotion Time Cure Model},
  author={Medina-Olivares, Victor and Lessmann, Stefan and Klein, Nadja},
  journal={arXiv preprint arXiv:2305.11575},
  year={2023}
}
```