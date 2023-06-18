```
├── dataset
    ├── A
    │   ├── A_B
    │   │   ├── test
    │   │   │   └── biwi_hotel.txt
    │   │   ├── train
    │   │   │   └── biwi_eth_train.txt
    │   │   └── val
    │   │       └── biwi_hotel_val.txt
    │   ├── ...
    ├── ...
├── experiments
    ├── test
    │   ├── 1
    │   │   ├── checkpoints
    │   │   ├── ...
    └── training
        ├── 1
        │   ├── checkpoints
        │       └── epoch_200.pth
        │   ├── config.json
        │   ├── cout.txt
        │   ├── loss_arr_train.npy
        │   ├── loss_arr_val.npy
        │   ├── metrics.json
        │   └── run.json
        ├── ...
├── t_gnn_lib
    ├── dataset.py
    ├── __init__.py
    ├── model.py
    ├── train_helpers.py
    └── utils.py
├── README.md
├── test_script.py
├── train_script.py
├── train_config.yml
```