from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.models import FTTransformer, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.training import Trainer
from sklearn.base import BaseEstimator, TransformerMixin

from sk_transformers.utils import check_ready_to_transform


class ToVecTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cat_embed_columns: List[str],
        continuous_columns: List[str],
        n_epochs: int = 1,
        batch_size: int = 32,
        input_dim: int = 32,
        n_heads: int = 4,
        n_blocks: int = 4,
        verbose: int = 1,
        preprocessing_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        training_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.cat_embed_columns = cat_embed_columns
        self.continuous_columns = continuous_columns
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
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: NDArray):

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
            "binary" if len(np.unique(y)) == 2 else "multiclass",
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

    def transform(self, X: pd.DataFrame):
        if self.tab_vec_ is None:
            raise ValueError("Transformer must be fitted first!")

        X = check_ready_to_transform(
            self,
            X,
            self.cat_embed_columns + self.continuous_columns,
        )
        return self.tab_vec_.transform(X)
