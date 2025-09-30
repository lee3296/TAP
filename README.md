# TAP: TWO-STAGE ADAPTIVE PERSONALIZATION OF MULTI-TASK AND MULTI-MODAL FOUNDATION MODELS IN FEDERATED LEARNING

## Installation

Firstly, download the necessary packages needed to run the implementation via requirements.txt.

```bash
pip install -r requirements.txt
```

In addition, note that all parts of the README assumes that the TAP implementation folder is under the home directory.

## Training
For training, as a first step, run the file torch_to_hf.py to convert the necessary torch datasets into HF format via

```bash
python3 ${HOME}/TAP_final/utils/torch_to_hf.py
```
and change the save paths in torch_to_hf.py to be compatiable with your enviornment, more specifically:
```python
fmnist_train = datasets.FashionMNIST(
    root='/path/to/HOME/TAP_final/data',
    train=True,
    download=True,
    transform=transform
)
```
```python
fmnist_hf.save_to_disk("/path/to/HOME/TAP_final/data/fmnist_hf")
```

Then, to begin training of the TAP algorithm, naviagate to the scripts directory and run run_train.sh.

```bash
bash ${HOME}/TAP_final/scripts/run_train.sh
```

Within run_train.sh, relevant settings can be changed as needed. When running TAP, FedAvg-based baselines will simulatenously be run and saved. To run any of the other baselines, change the TRAINING_ALGO field. 

NOTE: If you are running this script for the first time, you should comment out the following field in run_train.sh:

```bash
export HF_DATASETS_OFFLINE=1
```
and set it back to 1 afterwards to prevent execessive calls to the HF Hub.

## Evaluation

Similar to training, before running the evaluation script, make sure you change the file paths in hf_fromtorch_test.py, i.e.

```python
fmnist_test = datasets.FashionMNIST(
    root='/path/to/HOME/TAP_final/data',
    train=False,
    download=True,
    transform=transform
)
```
```python
fmnist_hf.save_to_disk("/path/to/HOME/TAP_final/data/fmnist_test")
```
and run 

```bash
python3 ${HOME}/TAP_final/evaluation/hf_fromtorch_test.py
```

Then, to evaluate the performance of the trained model, please run the run_eval.sh script under the file path below.

```bash
bash ${HOME}/TAP_final/scripts/run_eval.sh
```

Like the training script, fields in run_eval.sh can be changed based off the settings and algorithm being evaluted.

NOTE: Exactly like run_train.sh, make sure to turn off offline mode when running run_eval.sh for the first time to download the test dataset splits.
