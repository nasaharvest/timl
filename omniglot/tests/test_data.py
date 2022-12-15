from unittest import TestCase

from src.data import OmniglotAlphabet, ALPHABETS
from src.config import NUM_WAYS


class TestData(TestCase):
    def test_loads_correct(self):
        for alphabet in ALPHABETS:
            d = OmniglotAlphabet(alphabet)
            (t_x, t_y), (e_x, e_y) = d.sample_task(
                mode="train", train_samples_per_way=2, eval_samples_per_way=2
            )

            self.assertTrue(len(t_y.unique()) == NUM_WAYS)

            self.assertTrue(
                len(t_x) == len(t_y) == len(e_x) == len(e_y) == 2 * NUM_WAYS
            )
            self.assertTrue(t_x.shape == t_x.shape == (2 * NUM_WAYS, 1, 28, 28))
