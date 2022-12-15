import torch
from torchvision.transforms import functional as F
from torchvision import transforms

from pathlib import Path
from random import choice, sample, shuffle
import json

from PIL import Image
from PIL.Image import LANCZOS

from .config import NUM_WAYS

from typing import Tuple, List


OMNIGLOT = Path(__file__).parents[1] / "omniglot"
ALPHABETS = [x.name for x in list(OMNIGLOT.glob("*")) if x.name != ".DS_Store"]
VAL_TEST_SPLIT = json.load((Path(__file__).parent / "val_test_split.json").open())


class OmniglotAlphabet:
    def __init__(
        self,
        alphabet: str,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:

        assert alphabet in [x.name for x in OMNIGLOT.glob("*")]
        self.device = device

        self.alphabet = alphabet
        self.all_characters = [
            x.name for x in (OMNIGLOT / alphabet).glob("*") if x.name != ".DS_Store"
        ]

        self.data_transforms = transforms.Compose(
            [
                transforms.Resize(28, interpolation=LANCZOS),
                transforms.ToTensor(),
                lambda x: 1.0 - x,
            ]
        )

    @property
    def idx(self) -> int:
        return ALPHABETS.index(self.alphabet)

    def sample_characters(self, mode: str) -> List[str]:
        assert mode in ["train", "val", "test"]
        val_characters = VAL_TEST_SPLIT[self.alphabet]["val"]
        test_characters = VAL_TEST_SPLIT[self.alphabet]["test"]

        if mode == "val":
            return val_characters
        elif mode == "test":
            return test_characters
        else:
            train_characters = [
                x
                for x in self.all_characters
                if x not in val_characters + test_characters
            ]
            return sample(train_characters, NUM_WAYS)

    def _load_image(self, image_path: Path, rotate_amount: int = 0) -> torch.Tensor:
        image = Image.open(image_path, mode="r").convert("L")
        image = self.data_transforms(image)
        if rotate_amount != 0:
            image = F.rotate(image, angle=rotate_amount)
        return image

    def sample_task(
        self, mode: str, train_samples_per_way: int, eval_samples_per_way: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        characters_to_idx = {
            idx: character for idx, character in enumerate(self.sample_characters(mode))
        }

        t_x, t_y, e_x, e_y = [], [], [], []
        for char_idx, character in characters_to_idx.items():
            rotate_amount = 0
            if mode == "train":
                rotate_amount = choice([0, 90, 180, 270])
            all_image_paths = list((OMNIGLOT / self.alphabet / character).glob("*.png"))
            shuffle(all_image_paths)
            for idx in range(train_samples_per_way):
                t_y.append(char_idx)
                t_x.append(
                    self._load_image(all_image_paths[idx], rotate_amount=rotate_amount)
                )

            eval_images = all_image_paths[train_samples_per_way:]
            if eval_samples_per_way != -1:
                eval_images = eval_images[:eval_samples_per_way]

            for image in eval_images:
                e_y.append(char_idx)
                e_x.append(self._load_image(image, rotate_amount=rotate_amount))

        return (
            torch.stack(t_x).to(self.device),
            torch.tensor(t_y, device=self.device),
        ), (
            torch.stack(e_x).to(self.device),
            torch.tensor(e_y, device=self.device),
        )
