import pickle
import numpy as np
from .convmorph_dataset import (
    ConvmorphDatasetArc, ConvmorphDatasetOreo)

def split_train_test(split_path, cm_ds):
    with open(split_path, "rb") as fin:
        data = pickle.load(fin)
        train_idxs = data["train_idxs"]
        test_idxs = data["test_idxs"]
    
    train_ds = ConvmorphDatasetArc(cm_ds, train_idxs)
    test_ds = ConvmorphDatasetArc(cm_ds, test_idxs)
    train_eval_ds = ConvmorphDatasetArc(cm_ds, train_idxs[-len(test_ds):])
    

    return train_ds, test_ds, train_eval_ds, train_idxs, test_idxs

def split_train_test_oreo(split_path, cm_ds):
    with open(split_path, "rb") as fin:
        data = pickle.load(fin)
        train_idxs = data["train_idxs"]
        test_idxs = data["test_idxs"]
    
    train_ds = ConvmorphDatasetOreo(cm_ds, train_idxs)
    test_ds = ConvmorphDatasetOreo(cm_ds, test_idxs)
    train_eval_ds = ConvmorphDatasetOreo(cm_ds, train_idxs[-len(test_ds):])    

    return train_ds, test_ds, train_eval_ds, train_idxs, test_idxs
    