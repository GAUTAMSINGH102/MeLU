import datetime
import pandas as pd


class movielens_1m(object):
    def __init__(self):
        self.user_data, self.item_data, self.score_data = self.load()

    def load(self):
        path = "movielens/ml-1m"
        profile_data_path = "{}/users.dat".format(path)
        score_data_path = "{}/ratings.dat".format(path)
        item_data_path = "{}/products_extrainfos.dat".format(path)

        profile_data = pd.read_csv(
            profile_data_path, names=['user_id', 'city', 'zip'],
            sep=",", engine='python'
        )
        item_data = pd.read_csv(
            item_data_path, names=['Handle', 'Title', 'Custom Product Type', 'Tags', 'Variant SKU'],
            sep=",", engine='python', encoding="utf-8"
        )
        score_data = pd.read_csv(
            score_data_path, names=['user_id', 'product_id', 'rating'],
            sep=",", engine='python'
        )

        return profile_data, item_data, score_data
