import numpy as np
from sklearn.pipeline import make_pipeline

from sk_transformers.encoder_transformer import MeanEncoderTransformer

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_mean_encoder_in_pipeline(X_categorical, y_categorical) -> None:
    pipeline = make_pipeline(MeanEncoderTransformer())
    print()
    result = pipeline.fit_transform(X_categorical[["a"]], y_categorical)
    expected = np.array(
        [[0.4], [0.33333333], [0.33333333], [0.4], [0.4], [0.33333333], [0.4], [0.4]]
    )

    assert result.shape[0] == 8
    assert np.array_equal(result.to_numpy().round(4), expected.round(4))
    assert pipeline.steps[0][0] == "meanencodertransformer"
