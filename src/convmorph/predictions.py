import numpy as np
import torch

def make_predictions(model, word, kv, topk=5, metainfo=True, show_original=False):  
    model.eval()
    with torch.no_grad():
        const1_vec = kv[word[0]]
        const2_vec = kv[word[1]]
        inputX = np.concatenate([const1_vec, const2_vec]).reshape(20, 20)
        inputX = np.expand_dims(inputX, axis=(0, 1))
        batch_x = {"inputX": torch.tensor(inputX, dtype=torch.float32).to("cuda")}
        pred_vec = model(**batch_x).pred_vec.cpu().numpy().squeeze()
        distances = kv.distances(pred_vec)
        arg_idx = np.argsort(distances)
        word_idx = kv.key_to_index.get(word, -1)
        candidates = [(kv.index_to_key[idx], 1-distances[idx]) for idx in arg_idx[:topk]]

        if word_idx >= 0:
            pred_rank = np.where(arg_idx==word_idx)[0][0]
        else:
            pred_rank = -1

    if metainfo:
        print("In dataset: {}".format(
            word in kv.key_to_index))
        print("Predicted rank: ", pred_rank)

    if show_original and word_idx >= 0:
        print("-- In ori. embedding --")
        print(*kv.most_similar(word, topn=topk), sep="\n")
        print("-----------------------")
        
    return candidates
    