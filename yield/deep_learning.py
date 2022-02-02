from pathlib import Path


from config import (
    LSTM_CONFIGS,
    CNN_CONFIGS,
    ENCODER_VECTOR_SIZES,
    ENCODER_DROPOUT,
    DL_TAML,
)

from src import train_timl_model
from src.utils import download_and_extract_archive, HISTOGRAM_NAME


if __name__ == "__main__":

    data_folder = Path("data")
    checkpoint = True
    model_type = "cnn"

    download_and_extract_archive(data_folder, HISTOGRAM_NAME)

    for min_test_year in [2011, 2012, 2013, 2014, 2015, 2016]:

        model_name = f"{DL_TAML}_{min_test_year}"

        model = train_timl_model(
            data_folder,
            model_type=model_type,
            model_kwargs=LSTM_CONFIGS if model_type == "lstm" else CNN_CONFIGS,
            encoder_vector_sizes=ENCODER_VECTOR_SIZES,
            encoder_dropout=ENCODER_DROPOUT,
            model_name=model_name,
            min_test_year=min_test_year,
        )
        model.test()
