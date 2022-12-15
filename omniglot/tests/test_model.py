from unittest import TestCase

import torch

from src.data import OmniglotAlphabet, ALPHABETS
from src.config import NUM_WAYS
from src.model import OmniglotCNN, Encoder
from src.timl import Learner


class TestModel(TestCase):
    def test_forward_pass(self):
        model = OmniglotCNN()
        d = OmniglotAlphabet(ALPHABETS[0])
        (t_x, _), _ = d.sample_task(
            mode="train", train_samples_per_way=2, eval_samples_per_way=2
        )

        with torch.no_grad():
            output = model(t_x)
        self.assertTrue(output.shape == (NUM_WAYS * 2, NUM_WAYS))

    def test_timl_encoding(self):
        encoder = Encoder(method="timl")
        model = OmniglotCNN()

        d = OmniglotAlphabet(ALPHABETS[0])
        (t_x, _), _ = d.sample_task(
            mode="train", train_samples_per_way=2, eval_samples_per_way=2
        )

        with torch.no_grad():
            print(d.idx)
            embeddings = encoder(Learner.one_hot_from_idx(d.idx))
            model.update_embeddings(embeddings)
            output = model(t_x)
        self.assertTrue(output.shape == (NUM_WAYS * 2, NUM_WAYS))

    def test_mmaml_encoding(self):
        encoder = Encoder(method="mmaml")
        model = OmniglotCNN()

        d = OmniglotAlphabet(ALPHABETS[0])
        (t_x, t_y), _ = d.sample_task(
            mode="train", train_samples_per_way=2, eval_samples_per_way=2
        )

        with torch.no_grad():
            print(d.idx)
            embeddings = encoder((t_x, t_y))
            model.update_embeddings(embeddings)
            output = model(t_x)
        self.assertTrue(output.shape == (NUM_WAYS * 2, NUM_WAYS))
