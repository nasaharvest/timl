SHUFFLE_SEEDS = list(range(10))

# Model names
DL_TAML = "DL_TAML"

# LSTM model configurations
LSTM_CONFIGS = {
    "classifier_vector_inout": [128, 256, 1],
    "classifier_dropout": 0.2,
    "classifier_base_layers": 1,
}

CNN_CONFIGS = {
    "channel_sizes": [128, 256, 256, 512, 512, 512],
    "stride_list": [1, 2, 1, 2, 1, 2],
    "dropout": 0.5,
    "dense_features": [2048, 1],
}

ENCODER_VECTOR_SIZES = [32, 64, 128]
ENCODER_DROPOUT = 0.2
