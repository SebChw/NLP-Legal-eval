from typing import List, Tuple, Union

import numpy as np

def compute_class_weight(
    class_weight,
    classes: Union[np.ndarray, List[int], Tuple[int, ...]],
    y,
): ...
