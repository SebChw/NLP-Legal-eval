import numpy as np

from legal_eval.utils import get_class_counts


def test_get_class_counts(casted_dataset_dict):
    class_counts = get_class_counts(casted_dataset_dict["train"])
    # fmt: off
    np.array_equal(class_counts, np.array([1., 3., 2., 3., 81., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    # fmt: on
