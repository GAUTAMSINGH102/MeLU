import os
import torch
import pickle
import pandas as pd

from MeLU import MeLU
from options import config

from dataset import movielens_1m


def selection(melu, master_path, topk):
    if not os.path.exists("{}/scores/".format(master_path)):
        os.mkdir("{}/scores/".format(master_path))
    # if config['use_cuda']:
    #     melu.cuda()
    melu.eval()

    target_state = 'warm_state'
    dataset_size = int(len(os.listdir("{}/{}".format(master_path, target_state))) / 4)
    print(dataset_size)
    grad_norms = {}
    for j in list(range(dataset_size)):
        support_xs = pickle.load(open("{}/{}/supp_x_{}.pkl".format(master_path, target_state, j), "rb"))
        support_ys = pickle.load(open("{}/{}/supp_y_{}.pkl".format(master_path, target_state, j), "rb"))
        item_ids = []
        with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, target_state, j), "r") as f:
            for line in f.readlines():
                item_id = line.strip().split()[1]
                item_ids.append(item_id)

        # print(support_xs)
        # print(support_ys)
        # print(item_ids)

        for support_x, support_y, item_id in zip(support_xs, support_ys, item_ids):
            support_x = support_x.view(1, -1)
            support_y = support_y.view(1, -1)
            norm = melu.get_weight_avg_norm(support_x, support_y, config['inner'])
            try:
                grad_norms[item_id]['discriminative_value'] += norm.item()
                grad_norms[item_id]['popularity_value'] += 1
            except:
                grad_norms[item_id] = {
                    'discriminative_value': norm.item(),
                    'popularity_value': 1
                }

    d_value_max = 0
    p_value_max = 0
    for item_id in grad_norms.keys():
        grad_norms[item_id]['discriminative_value'] /= grad_norms[item_id]['popularity_value']
        if grad_norms[item_id]['discriminative_value'] > d_value_max:
            d_value_max = grad_norms[item_id]['discriminative_value']
        if grad_norms[item_id]['popularity_value'] > p_value_max:
            p_value_max = grad_norms[item_id]['popularity_value']
    for item_id in grad_norms.keys():
        grad_norms[item_id]['discriminative_value'] /= float(d_value_max)
        grad_norms[item_id]['popularity_value'] /= float(p_value_max)
        grad_norms[item_id]['final_score'] = grad_norms[item_id]['discriminative_value'] * grad_norms[item_id]['popularity_value']

    dataset = movielens_1m()

    product_info = {}

    df = pd.read_csv("movielens/ml-1m/products_extrainfo.txt", dtype={"Variant SKU": "string"})
    for idx, row in df.iterrows():
        sku = row['Variant SKU']
        title = row['Title']
        product_info[sku] = "{} ({})".format(title, sku)
        # print(product_info)
        # print(row)

    evidence_candidates = []
    for item_id, value in list(sorted(grad_norms.items(), key=lambda x: x[1]['final_score'], reverse=True))[:topk]:
        evidence_candidates.append((product_info[item_id], value['final_score']))
    return evidence_candidates
