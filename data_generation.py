import re
import os
import json
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm

from options import states
from dataset import movielens_1m


def item_converting(row, tags_list, type_list):
    tags_idx = torch.zeros(1, 82).long()
    for tag in str(row['Tags']).split(", "):
        idx = tags_list.index(tag)
        tags_idx[0, idx] = 1

    type_idx = torch.tensor([[type_list.index(str(row['Custom Product Type']))]]).long()

    return torch.cat((tags_idx, type_idx), 1)


def user_converting(row, city_list, zipcode_list):
    city_idx = torch.tensor([[city_list.index(str(row['city']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:6])]]).long()
    return torch.cat((city_idx, zip_idx), 1)


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


def generate(master_path):
    dataset_path = "movielens/ml-1m"
    tags_list = load_list("{}/tags.txt".format(dataset_path))
    type_list = load_list("{}/type.txt".format(dataset_path))
    city_list = load_list("{}/city.txt".format(dataset_path))
    zipcode_list = load_list("{}/zipcode.txt".format(dataset_path))

    if not os.path.exists("{}/warm_state/".format(master_path)):
        for state in states:
            os.mkdir("{}/{}/".format(master_path, state))
    if not os.path.exists("{}/log/".format(master_path)):
        os.mkdir("{}/log/".format(master_path))

    dataset = movielens_1m()

    # hashmap for item information
    if not os.path.exists("{}/m_product_dict.pkl".format(master_path)):
        product_dict = {}
        for idx, row in dataset.item_data.iterrows():
            m_info = item_converting(row, tags_list, type_list)
            product_dict[row['Variant SKU']] = m_info
        pickle.dump(product_dict, open("{}/m_product_dict.pkl".format(master_path), "wb"))
    else:
        product_dict = pickle.load(open("{}/m_product_dict.pkl".format(master_path), "rb"))

    # hashmap for user profile
    if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
        user_dict = {}
        for idx, row in dataset.user_data.iterrows():
            u_info = user_converting(row, city_list, zipcode_list)
            user_dict[row['user_id']] = u_info
        pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
    else:
        user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

    for state in states:
        idx = 0
        if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            os.mkdir("{}/{}/{}".format(master_path, "log", state))
        with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())
        for _, user_id in (enumerate(dataset.keys())):
            u_id = int(user_id)
            seen_product_len = len(dataset[str(u_id)])
            indices = list(range(seen_product_len))

            random.shuffle(indices)
            tmp_x = np.array(dataset[str(u_id)])
            tmp_y = np.array(dataset_y[str(u_id)])

            support_x_app = None
            for p_id in tmp_x[indices[:]]:
                p_id = int(p_id)
                tmp_x_converted = torch.cat((product_dict[p_id], user_dict[u_id]), 1)
                try:
                    support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                except:
                    support_x_app = tmp_x_converted

            query_x_app = None
            for p_id in tmp_x[indices[:]]:
                p_id = int(p_id)
                u_id = int(user_id)
                tmp_x_converted = torch.cat((product_dict[p_id], user_dict[u_id]), 1)
                try:
                    query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                except:
                    query_x_app = tmp_x_converted
            support_y_app = torch.FloatTensor(tmp_y[indices[:]])
            query_y_app = torch.FloatTensor(tmp_y[indices[:]])

            pickle.dump(support_x_app, open("{}/{}/supp_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "wb"))
            with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[:]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[:]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            idx += 1

# master_path= "./ml"
# generate(master_path)
