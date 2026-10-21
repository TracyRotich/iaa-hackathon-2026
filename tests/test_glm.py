import numpy as np
import pandas as pd
import pytest

from iaa.glm import CredibilityEncoder, buhlmann_kappa
from iaa.metrics import normalised_gini


def test_toy_example_kappa_20():
    # worked example: categories A,B,C, kappa = 20, global mean 2.44
    y = np.array([2.0, 1.0, 3.0, 2.0, 0.5, 4.0, 2.0, 4.0, 3.5]); g = list("ABABCABCC"); w = np.ones(9)
    enc = CredibilityEncoder(kappa=20).fit(g, y, w, process_var=1.0)
    assert enc.transform(["A"])[0] == pytest.approx(3 / 23 * 3.0 + 20 / 23 * y.mean(), abs=1e-3)  # 2.513
    loo = enc.transform_train(g, y, w)
    assert loo[0] == pytest.approx(2 / 22 * 3.5 + 20 / 22 * y.mean(), abs=1e-3)  # 2.536 in the slide


def test_kappa_large_when_groups_identical():
    assert buhlmann_kappa([10, 10, 10], [100, 100, 100], process_var=0.1) == np.inf


def test_gini_perfect_and_random():
    rng = np.random.default_rng(0); y = rng.gamma(1, 1, 2000); w = np.ones(2000)
    assert normalised_gini(y, w, y) == pytest.approx(1.0)
    assert abs(normalised_gini(y, w, rng.random(2000))) < 0.1
