from src.data.yield_data import CropYieldDataset


def test_latlons():
    ds = CropYieldDataset()

    region_tuples = ds.region_tuples
    for _, row in region_tuples.iterrows():
        _ = ds.region_to_latlon(row.loc1, row.loc2)
