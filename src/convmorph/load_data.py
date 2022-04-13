import pickle
import numpy as np
from .convmorph_dataset import ConvmorphDatasetArc

def split_train_test(split_path, cm_ds):
    with open(split_path, "rb") as fin:
        data = pickle.load(fin)
        train_idxs = data["train_idxs"]
        test_idxs = data["test_idxs"]
    N = len(cm_ds)
    rng = np.random.RandomState(123)
    random_split = np.arange(N)
    rng.shuffle(random_split)
    # train_words = set(cm_ds[i]["word"] for i in train_idxs)
    train_ds = ConvmorphDatasetArc(cm_ds, train_idxs)
    test_ds = ConvmorphDatasetArc(cm_ds, test_idxs)
    train_eval_ds = ConvmorphDatasetArc(cm_ds, train_idxs[-len(test_ds):])
    

    return train_ds, test_ds, train_eval_ds, train_idxs, test_idxs
    