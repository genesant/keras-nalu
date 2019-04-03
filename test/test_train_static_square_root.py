"""Test training on a static square root task"""

from test.shared import train
import numpy as np

def test_train_static_square_root():
    """Test training on a static square root task"""

    _, interpolation_loss, extrapolation_loss = train(
        epoch_count=4000,
        learning_rate=0.05,
        task=lambda a, _: np.sqrt(a),
    )

    assert interpolation_loss < 0.0001
    assert extrapolation_loss < 0.0001
