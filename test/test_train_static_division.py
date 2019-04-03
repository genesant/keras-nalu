"""Test training on a static division task"""

from test.shared import train_retry

def test_train_static_division():
    """Test training on a static division task"""

    train_retry(
        epoch_count=10000,
        expected_interpolation_loss=0.0625,
        expected_extrapolation_loss=0.2920,
        learning_rate=0.002,
        task=lambda a, b: a / b,
    )
