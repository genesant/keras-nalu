"""Test loading and predicting with an NALU model"""

from tempfile import mkstemp
from test.shared import generate_dataset, train_retry
from keras.models import load_model

def test_load_and_predict():
    """Test loading and predicting with an NALU model"""

    task = lambda a, _: a

    model, _, _ = train_retry(
        epoch_count=1000,
        expected_extrapolation_loss=0.0001,
        expected_interpolation_loss=0.0001,
        learning_rate=0.05,
        task=task,
    )

    _, model_path = mkstemp(suffix='.h5')
    model.save(model_path)
    model = load_model(model_path)

    dataset = generate_dataset(size=8, task=task)
    X = dataset['X_test']
    Y = dataset['Y_test']
    Y_hat = model.predict(X)
    loss = ((Y - Y_hat) ** 2).mean()

    assert loss < 0.0001
