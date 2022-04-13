import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

class ConvmorphDataset(Dataset):
    def __init__(self, vocabs, embs):
        self.build_dataset(vocabs, embs)

    def build_dataset(self, vocabs, embs):
        vocab_map = {vocab: idx for idx, vocab in enumerate(vocabs)}
        self.data = []
        for idx in tqdm(range(len(vocabs)), desc="building dataset"):
            word = vocabs[idx]
            if len(word) < 2: continue
            if not (word[0] in vocab_map and word[1] in vocab_map):
                continue
        emb = embs[idx]
        const1_vec = embs[vocab_map[word[0]]]
        const2_vec = embs[vocab_map[word[1]]]
        const1 = word[0]
        const2 = word[1]
        self.data.append(dict(
            word=word,
            const1=const1, const2=const2,
            word_vec=emb,
            const1_vec=const1_vec, const2_vec=const2_vec
        ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ConvmorphNNDataset(Dataset):
    def __init__(self, vocabs, embs):
        self.build_dataset(vocabs, embs)

    def build_dataset(self, vocabs, embs):
        vocab_map = {vocab: idx for idx, vocab in enumerate(vocabs)}
        self.data = []
        for idx in tqdm(range(len(vocabs)), desc="building dataset"):
            word = vocabs[idx]
            const1 = word[:2]
            const2 = word[2:]
            if len(word) < 4: continue
            if not (const1 in vocab_map and const2 in vocab_map):
                continue
            emb = embs[idx]
            const1_vec = embs[vocab_map[const2]]
            const2_vec = embs[vocab_map[const1]]      
            self.data.append(dict(
                word=word,
                const1=const1, const2=const2,
                word_vec=emb,
                const1_vec=const1_vec, const2_vec=const2_vec
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ConvmorphDatasetArc(Dataset):
    def __init__(self, cm_dataset, idxs):
        self.build_dataset(cm_dataset, idxs)

    def build_dataset(self, ds, idxs):
        self.data = []
        for serial, idx in enumerate(idxs):
            data_x = ds[idx]
            inputX = np.concatenate([
                data_x["const1_vec"], data_x["const2_vec"]
            ]).reshape(1, 20, 20)
            target = data_x["word_vec"]
            self.data.append(dict(
                word_id=serial,
                inputX=torch.tensor(inputX, dtype=torch.float32),
                target=torch.tensor(target, dtype=torch.float32)
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
