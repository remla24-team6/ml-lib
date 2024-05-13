# REMLA Team 6's ML Preprocessing Library
This is the implementation of the phishing detection pre-processing repository for CS4295 Release Engineering for Machine Learning Applications (Team 6) at TU Delft (Q4, 2024). 
This package supports the preprocessing (tokenisation and label encoding) for the [phishing detection dataset](https://www.kaggle.com/code/luiscruz/phishing-detection-cnn/input?scriptVersionId=173138322) so that the same logic can be re-used at both training and inference time. In order to support this, the library is packaged with a tokenizer and label encoder that is fit on the entire dataset. Please refer to the `PreprocessingUtil` class [here](/ml_lib_remla/preprocessing.py) for more information on how the tokenizer and label encoder are created.

The package release workflow is automatically triggered when a new Git tag is pushed. We follow semantic versioning for the library in the format `v<major>.<minor>.<patch>`.
The package repository (PyPi) can be found [here](https://pypi.org/project/ml-lib-remla/).


# Installation
```console
pip install ml-lib-remla
```

# Usage
To get the current version using this package, execute the following lines in Python -
```python 
from ml_lib_remla.preprocessing import Preprocessing
pp = Preprocessing()
print(pp.tokenize_single("www.google.com"))
```

The `Preprocessing` class supports the following functionalities -
- **tokenize_batch**: Performs the tokenization on a list of urls. This can be used to pre-process the entire dataset.
- **tokenize_single**: Performs the tokenization for a single url. This can be used to pre-process a single url at inference time.
- **encode_label_batch**: Performs the label encoding for a list of labels. This can be used to pre-process the target labels of the entire dataset.