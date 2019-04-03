"""Test training on a static subtraction task"""

from test.shared import train_retry

def test_train_static_subtraction():
    """Test training on a static subtraction task"""

    train_retry(
        epoch_count=2000,
        expected_interpolation_loss=0.0001,
        expected_extrapolation_loss=0.0001,
        learning_rate=0.05,
        task=lambda a, b: a - b,
    )
