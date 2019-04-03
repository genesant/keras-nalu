"""Test training on a static square task"""

from test.shared import train

def test_train_static_square():
    """Test training on a static square task"""

    _, interpolation_loss, extrapolation_loss = train(
        epoch_count=4000,
        learning_rate=0.05,
        task=lambda a, _: a ** 2,
    )

    assert interpolation_loss < 0.0001
    assert extrapolation_loss < 0.0001
