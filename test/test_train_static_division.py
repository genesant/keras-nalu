"""Test training on a static division task"""

from test.shared import train

def test_train_static_division():
    """Test training on a static division task"""

    _, interpolation_loss, extrapolation_loss = train(
        epoch_count=10000,
        learning_rate=0.002,
        task=lambda a, b: a * b,
    )

    assert interpolation_loss < 0.0625
    assert extrapolation_loss < 0.2920
