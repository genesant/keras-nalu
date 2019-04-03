"""Test training on a static addition task"""

from test.shared import train_retry

def test_train_static_addition():
    """Test training on a static addition task"""

    train_retry(
        epoch_count=1000,
        expected_interpolation_loss=0.0001,
        expected_extrapolation_loss=0.0012,
        learning_rate=0.05,
        task=lambda a, b: a + b,
    )
