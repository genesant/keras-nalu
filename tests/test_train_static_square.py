"""Test training on a static square task"""

from tests.shared import train_retry

def test_train_static_square():
    """Test training on a static square task"""

    train_retry(
        epoch_count=4000,
        expected_interpolation_loss=0.0001,
        expected_extrapolation_loss=0.0001,
        learning_rate=0.05,
        task=lambda a, _: a ** 2,
    )
