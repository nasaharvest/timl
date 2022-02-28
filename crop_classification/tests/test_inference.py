from datetime import datetime
from pathlib import Path
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import Inference

test_tif = Path(__file__).parent / "test_data/373-croplands_2016-02-07_2017-02-01.tif"


def test_start_date_from_str():
    actual_start_date = Inference.start_date_from_str(test_tif.name)
    expected_start_date = datetime(2016, 2, 7, 0, 0)
    assert actual_start_date == expected_start_date


def test_tif_to_np():
    x_np, flat_lat, flat_lon = Inference._tif_to_np(
        test_tif, start_date=datetime(2016, 2, 7, 0, 0)
    )
    assert x_np.shape, (289, 24, 18)
    assert flat_lat.shape, (289,)
    assert flat_lon.shape, (289,)


def test_combine_predictions():
    flat_lat = np.array(
        [14.95313164, 14.95313164, 14.95313164, 14.95313164, 14.95313164]
    )
    flat_lon = np.array(
        [-86.25070894, -86.25061911, -86.25052928, -86.25043945, -86.25034962]
    )
    batch_predictions = np.array(
        [[0.43200156], [0.55286014], [0.5265], [0.5236109], [0.4110847]]
    )
    xr_predictions = Inference._combine_predictions(
        flat_lat=flat_lat, flat_lon=flat_lon, batch_predictions=batch_predictions
    )

    # Check size
    assert xr_predictions.dims["lat"], 1
    assert xr_predictions.dims["lon"], 5

    # Check coords
    assert (xr_predictions.lat.values == flat_lat[0:1]).all()
    assert (xr_predictions.lon.values == flat_lon).all()

    # Check all predictions between 0 and 1
    assert xr_predictions.min() >= 0
    assert xr_predictions.max() <= 1
