from sklearn.pipeline import make_pipeline

from sk_transformers.deep_transformer import ToVecTransformer

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_to_vec_transformer_in_pipeline(adult_dataframe):
    df = adult_dataframe[:1000]
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

    pipeline = make_pipeline(
        ToVecTransformer(
            cat_cols,
            cont_cols,
            training_objective="binary",
            verbose=0,
            n_epochs=1,
            input_dim=2,
            n_blocks=1,
            n_heads=1,
        )
    )
    assert pipeline.fit_transform(df, target).shape == (1000, 26)
    assert pipeline.steps[0][0] == "tovectransformer"
    assert pipeline.steps[0][1].tab_vec_.__class__.__name__ == "Tab2Vec"
