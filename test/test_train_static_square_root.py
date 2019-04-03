"""Test training on a static square root task"""

from test.shared import train_retry
import numpy as np

def test_train_static_square_root():
    """Test training on a static square root task"""

    train_retry(
        epoch_count=4000,
        expected_interpolation_loss=0.0001,
        expected_extrapolation_loss=0.0001,
        learning_rate=0.05,
        task=lambda a, _: np.sqrt(a),
    )
