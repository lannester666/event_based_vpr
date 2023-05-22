import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable


def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?'):
        # add precision at k
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
        # build index
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)
        
        # add references
        faiss_index.add(r_list)

        # search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))
        
        
        
        # start calculating recall_at_k
        correct_at_k = np.zeros(len(k_values))
        correct_at_k_recall = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        places = 15
        gt_places = [0] * places
        pred_places = [0] * places
        precision = [0] * places
        # import pdb; pdb.set_trace()
        for i in range(predictions.shape[0]):
            gt_places[gt[i]] = gt_places[gt[i]] + 1
            if predictions[i][0] == gt[i]:
                pred_places[predictions[i][0]] = pred_places[predictions[i][0]] + 1
        for i in range(len(gt_places)):
            precision[i] = pred_places[i]/gt_places[i]
        precision = np.mean(precision)
        # import torch
        # import pdb; pdb.set_trace()
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print() # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values] + ['Precision'] + ['Precision@1']
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k] + ['Precision']+ [f'{100*precision:.2f}'])
            print(table.get_string(title=f"Performances on {dataset_name}"))
        
        return d, precision
