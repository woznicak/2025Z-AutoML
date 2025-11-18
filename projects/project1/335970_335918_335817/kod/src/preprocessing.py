import numpy as np
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def preprocess_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    le = LabelEncoder()
    y = le.fit_transform(y)

    num_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer())
    ])

    cat_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('one_hot', OneHotEncoder(handle_unknown='ignore',
         drop='first', sparse_output=False))
    ])

    col_trans = ColumnTransformer(
        transformers=[
            (
                "numeric_preprocessing",
                num_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical_preprocessing",
                cat_pipeline,
                make_column_selector(dtype_include=['object', 'category']),
            ),
        ],
        remainder="passthrough",
    )

    return X, y, col_trans
