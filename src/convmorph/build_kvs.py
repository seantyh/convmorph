import numpy as np
from tqdm.auto import tqdm
from gensim.models import KeyedVectors

def build_concat_space(vocabs, embs):
    kv = KeyedVectors(200)
    kv.add_vectors(vocabs, embs)
    vocab1 = []; mask = []; embs1 = []
    for word, emb_x in tqdm(zip(vocabs, embs)):
        if len(word) < 2:
            mask.append(False) 
            continue
        if word[0] not in vocabs or word[1] not in vocabs:
            mask.append(False)
            continue
        vocab1.append(word)
        mask.append(True)
        embs1.append(np.concatenate([kv[word[0]], kv[word[1]]]))
    embs1 = np.vstack(embs1)
    embs2 = embs[mask, :]

    # embs1: n_vocab1 x 400
    kv_const = KeyedVectors(embs1.shape[1])
    kv_const.add_vectors(vocab1, embs1)

    # embs2: n_vocab2 x 200
    kv_word = KeyedVectors(200)
    vocab2 = [x for x, m in zip(vocabs, mask) if m]
    kv_word.add_vectors(vocab2, embs2)

    return {"kv_const": kv_const, "kv_word": kv_word}