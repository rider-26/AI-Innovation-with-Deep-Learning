import numpy as np
import pandas as pd
import pytest

from shopsphere.split import assign_split


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    n = 10_000
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "target": (rng.random(n) < 0.146).astype(int),  # same class balance as train.csv
        }
    )


def test_fractions_and_no_overlap(labels):
    out = assign_split(labels)
    shares = out["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.001)
    assert shares["val"] == pytest.approx(0.15, abs=0.001)
    assert shares["test"] == pytest.approx(0.15, abs=0.001)
    assert out["split"].notna().all()
    assert len(out) == len(labels)


def test_stratified_on_target(labels):
    out = assign_split(labels)
    overall = labels["target"].mean()
    for _, part in out.groupby("split"):
        assert part["target"].mean() == pytest.approx(overall, abs=0.002)


def test_same_split_whatever_the_row_order(labels):
    a = assign_split(labels).set_index("customer_id")["split"]
    b = assign_split(labels.sample(frac=1, random_state=1)).set_index("customer_id")["split"]
    assert a.sort_index().equals(b.sort_index())


def test_duplicate_ids_rejected(labels):
    with pytest.raises(ValueError):
        assign_split(pd.concat([labels, labels.head(1)]))
