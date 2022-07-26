import torch
import torch.nn as nn
import torch.nn.functional as F


class item(torch.nn.Module):
    def __init__(self, config):
        super(item, self).__init__()
        self.num_tag = config['num_tag']
        self.num_type = config['num_type']
        self.embedding_dim = config['embedding_dim']
        
        self.embedding_tag = torch.nn.Linear(
            in_features=self.num_tag,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_type = torch.nn.Embedding(
            num_embeddings=self.num_type,
            embedding_dim=self.embedding_dim
        )

    def forward(self, tag_idx, type_idx, vars=None):
        tag_emb = self.embedding_tag(tag_idx.float()) / torch.sum(tag_idx.float(), 1).view(-1, 1)
        type_emb = self.embedding_type(type_idx)
        return torch.cat((tag_emb, type_emb), 1)


class user(torch.nn.Module):
    def __init__(self, config):
        super(user, self).__init__()
        self.num_city = config['num_city']
        self.num_zipcode = config['num_zipcode']
        self.embedding_dim = config['embedding_dim']

        self.embedding_city = torch.nn.Embedding(
            num_embeddings=self.num_city,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, city_idx, area_idx):
        city_emb = self.embedding_city(city_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((city_emb, area_emb), 1)
