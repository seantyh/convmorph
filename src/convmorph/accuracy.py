import numpy as np
import torch
from tqdm.auto import tqdm

def compute_accuracy(model, data_loader, test_embs, topk=5, return_corrects=False):
    n_correct = 0
    n_items = 0
    model.eval()
    corrects = []
    for batch_x in tqdm(data_loader):
        with torch.no_grad():
            batch_x = {k: v.to("cuda") for k, v in batch_x.items() if k!="word"}
            word_ids = batch_x["word_id"].cpu().numpy()
            pred_vec = model(**batch_x).pred_vec         
            norm_vec = pred_vec / torch.norm(pred_vec, dim=1).unsqueeze(1)           
            preds = torch.argsort(-torch.matmul(norm_vec, test_embs.transpose(1, 0)), dim=1).cpu().numpy()[:, :topk]                
            in_topk = np.any(word_ids[:, np.newaxis]==preds, axis=1)        
            n_correct += np.array(in_topk, dtype=np.int32).sum()
            corrects.extend(word_ids[in_topk].tolist())
            n_items += len(preds)         

    if return_corrects:
        out = (n_correct/n_items, corrects)
    else:
        out = n_correct/n_items
    return out