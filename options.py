config = {
    # item
    'num_tag': 82,
    'num_type': 9,
    'embedding_dim': 32,
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,
    # user
    'num_city': 30,
    'num_zipcode': 42,
    # cuda setting
    'use_cuda': False,
    # model setting
    'inner': 1,
    'lr': 5e-5,
    'local_lr': 5e-6,
    'batch_size': 32,
    'num_epoch': 20,
    # candidate selection
    'num_candidate': 20,
}

states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]
