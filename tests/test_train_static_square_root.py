"""Test training on a static square root task"""

import numpy as np
from tests.shared import train_retry

def test_train_static_square_root():
    """Test training on a static square root task"""

    train_retry(
        epoch_count=4000,
        expected_interpolation_loss=0.0001,
        expected_extrapolation_loss=0.0001,
        learning_rate=0.05,
        task=lambda a, _: np.sqrt(a),
    )
