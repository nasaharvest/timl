import json
from pathlib import Path

from src.timl import train_timl_model
from src.utils import seed_everything


def main(awareness: str = "timl", seed: int = 42):
    seed_everything(seed)
    model = train_timl_model(awareness=awareness, seed=seed)
    outputs = model.test()
    json.dump(outputs, Path(model.model_folder / "test_results.json").open("w"))


if __name__ == "__main__":
    main("timl")
    main("mmaml")
    main(None)
