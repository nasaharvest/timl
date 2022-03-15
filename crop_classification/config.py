from typing import Dict, List


SHUFFLE_SEEDS = list(range(10))

DATASET_TO_SIZES: Dict[str, List] = {
    "Kenya_1_maize": [20, 32, 64, 96, 128, 160, 192, 224, 256, None],
    "Brazil_0_coffee": [None],
    "Togo_crop": [20, 50, 126, 254, 382, 508, 636, 764, 892, 1020, 1148, None],
}


# Model names
DL_TIML = "DL_TIML"
