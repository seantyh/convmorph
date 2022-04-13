
import gzip
import numpy as np
from tqdm.auto import tqdm

def load_tencent_compound(emb_path):
    vocabs = []
    embs = []

    with gzip.open(emb_path, "rt", encoding="UTF-8") as fin:
        n_vocab, n_hdim = fin.readline().split(" ")
        n_hdim = int(n_hdim)
        for i in tqdm(range(int(n_vocab))):
            toks = fin.readline().strip().split(" ")
            word = toks[0]
            emb = np.array([float(x) for x in toks[1:]])
            vocabs.append(word)
            embs.append(emb)
        embs = embs / np.linalg.norm(embs, axis=1)[:, np.newaxis]
    return vocabs, embs