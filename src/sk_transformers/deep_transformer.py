from typing import Any, Dict, List, Optional

import pandas as pd
from numpy.typing import NDArray
from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.models import FTTransformer, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.training import Trainer
from sklearn.base import BaseEstimator, TransformerMixin

from sk_transformers.utils import check_ready_to_transform


class ToVecTransformer(BaseEstimator, TransformerMixin):
    """This transformer trains an [FT-
    Transformer](https://paperswithcode.com/method/ft-transformer) using the.

    [pytorch-widedeep package](https://github.com/jrzaurin/pytorch-widedeep)
    and extracts the embeddings.

    from its embedding layer. The output shape of the transformer is (number of rows,(`input_dim` * number of columns)).
    Please refer to [this example](https://pytorch-widedeep.readthedocs.io/en/latest/examples/09_extracting_embeddings.html)
    for pytorch_widedeep example on how to extract embeddings.

    Example:
    ```python
    import numpy as np
    import pandas as pd
    from pytorch_widedeep.datasets import load_adult
    from sk_transformers import ToVecTransformer

    df = load_adult(as_frame=True)
    df["target"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df = df.drop(["income", "educational-num"], axis=1)

    cat_cols, cont_cols = [], []
    for col in df.columns:
        if df[col].dtype == "O" or df[col].nunique() < 50 and col != "target":
            cat_cols.append(col)
        elif col != "target":
            cont_cols.append(col)

    target_col = "target"
    target = df[target_col].to_numpy()

    transformer = ToVecTransformer(cat_cols, cont_cols, training_objective="binary")
    transformer.fit_transform(df, target).shape
    ```
    ```
    (48842, 416)
    ```

    Args:
        cat_embed_columns (List[str]): List of categorical columns to be embedded.
        continuous_columns (List[str]): List of continuous columns.
        training_objective (str): The training objective. Possible values are:
            Possible values are: binary, binary_focal_loss, multiclass, multiclass_focal_loss,
            regression, mean_absolute_error, mean_squared_log_error, root_mean_squared_error,
            root_mean_squared_log_error, zero_inflated_lognormal, quantile, tweedie.
            Read more here: https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/trainer.html#pytorch_widedeep.training.Trainer

        n_epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size to train the model.
        input_dim (int): The so-called *dimension of the model*.
            Is the number of embeddings used to encode the categorical and/or continuous columns.
        n_blocks (int): Number of FT-Transformer blocks.
        n_heads (int): Number of attention heads per FT-Transformer block.
        verbose (int): Verbosity level.
        preprocessing_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to the [`TabPreprocessor`](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/preprocessing.html#pytorch_widedeep.preprocessing.tab_preprocessor.TabPreprocessor).
        model_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to the [`FTTransformer`](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/model_components.html#pytorch_widedeep.models.tabular.transformers.ft_transformer.FTTransformer).
        training_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to the [`Trainer`](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/trainer.html#pytorch_widedeep.training.Trainer).
    """

    def __init__(
        self,
        cat_embed_columns: List[str],
        continuous_columns: List[str],
        training_objective: str,
        n_epochs: int = 1,
        batch_size: int = 32,
        input_dim: int = 32,
        n_blocks: int = 4,
        n_heads: int = 4,
        verbose: int = 1,
        preprocessing_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        training_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.cat_embed_columns = cat_embed_columns
        self.continuous_columns = continuous_columns
        self.training_objective = training_objective
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.verbose = verbose
        self.preprocessing_kwargs = preprocessing_kwargs
        self.training_kwargs = model_kwargs
        self.training_kwargs = training_kwargs
        self.tab_vec_: Optional[Tab2Vec] = None

    def fit(self, X: pd.DataFrame, y: NDArray) -> "ToVecTransformer":
        """Fits the `ToVecTransformer`. The `TabPreprocessor` is fitted and the
        `FTTransformer` is trained.

        Args:
            X (pd.DataFrame): The input data.
            y (NDArray): The target data.

        Returns:
            ToVecTransformer: The fitted transformer.
        """

        self.fitted_ = True

        preprocessor = TabPreprocessor(
            cat_embed_cols=self.cat_embed_columns,
            continuous_cols=self.continuous_columns,
            verbose=self.verbose,
            **self.preprocessing_kwargs or {},
        )

        preprocessor.fit_transform(X)

        ft_transformer = FTTransformer(
            column_idx=preprocessor.column_idx,
            cat_embed_input=preprocessor.cat_embed_input,
            continuous_cols=preprocessor.continuous_cols,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            input_dim=self.input_dim,
            **self.training_kwargs or {},
        )

        model = WideDeep(deeptabular=ft_transformer)

        trainer = Trainer(
            model,
            self.training_objective,
            seed=42,
            verbose=self.verbose,
            **self.training_kwargs or {},
        )

        trainer.fit(
            X_tab=preprocessor.fit_transform(X),
            target=y,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
        )
        self.tab_vec_ = Tab2Vec(model, preprocessor, return_dataframe=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input data and returns the embeddings.

        The output shape is (number of rows,(`input_dim` * number of columns)).

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The embeddings.
        """

        X = check_ready_to_transform(
            self,
            X,
            list(self.cat_embed_columns) + list(self.continuous_columns),
        )
        return self.tab_vec_.transform(X)  # type: ignore
