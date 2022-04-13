import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors

def compute_projections(model, words, ref_kv, device="cuda"):  
    model.eval()
    with torch.no_grad():
        neigh_vecs = np.vstack([ref_kv[word_x] for word_x in words])
        inputX = neigh_vecs.reshape(-1, 20, 20)
        inputX = np.expand_dims(inputX, axis=1)
        inputX = torch.tensor(inputX)
        pred_vecs = []
        for batch_x in DataLoader(TensorDataset(inputX), 
                                  shuffle=False, 
                                  batch_size=128):
            inputX_batch = batch_x[0].to(device)
            pred_batch = model(inputX=inputX_batch).pred_vec.cpu().numpy().squeeze()
            pred_vecs.append(pred_batch)
  
        pred_vecs = np.vstack(pred_vecs)
      
    return pred_vecs

def compute_neighbor_similarities(model, target, neighbors, target_kv: KeyedVectors, device="cuda"):
    target_vec = compute_projections(model, [target], target_kv, device)[0]
    neighbor_vecs = compute_projections(model, neighbors, target_kv, device)
    sim_vecs = target_kv.cosine_similarities(target_vec, neighbor_vecs)
    return sim_vecs

# ref_kv: kv2 (compound/word vectors)
# target_kv: kv1 (concated const. vectors)
def compute_spearmans(target, ref_kv, target_kv, model, linproj, 
                      topn=[10,20,30,50,80,100,150,200,300,400]):
    sim_rows = []
    for k in topn:
        ref_neighs = ref_kv.most_similar(target, topn=k)
        ref_words, ref_sims = list(zip(*ref_neighs))
        kv1_sims = 1-target_kv.distances(target, ref_words)
        conv_sims = compute_neighbor_similarities(model, target, ref_words, target_kv)
        linear_sims = compute_neighbor_similarities(linproj, target, ref_words, target_kv)
        sim_df = pd.DataFrame(dict(
                    word=ref_words, ref=ref_sims, 
                    kv1=kv1_sims, conv=conv_sims, lin=linear_sims))
        sim_corr = sim_df.corr(method="spearman").iloc[0,:]
        sim_corr["k"] = k
        sim_rows.append(sim_corr)
        corr_df = pd.DataFrame(sim_rows).reset_index(drop=True).drop("ref", axis=1)
        return corr_df