from __future__ import annotations
import re
from copy import deepcopy
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (FunctionTransformer, OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler,
                                    RobustScaler)
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import TransformerMixin


class OrdinalEncoderMaxUnknownValue(OrdinalEncoder):
    def __init__(
        self,
        *,
        categories="auto",
        dtype=np.float64,
        encoded_missing_value=np.nan,
        min_frequency=None,
        max_categories=None,
    ):
        super().__init__(
            categories=categories,
            dtype=dtype,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=encoded_missing_value,
            min_frequency=min_frequency,
            max_categories=max_categories,
        )
        self.unknown_values = None

    def fit(self, X, y=None):
        super().fit(X, y)
        # Store the number of categories for each feature, which will be used as the unknown value for each feature.
        # This makes that each unknown value is the next integer after the last category.
        self.unknown_values = [len(categories) for categories in self.categories_]
        return self

    def transform(self, X):
        X = super().transform(X)
        # Replace unknown values with the unknown value for each feature.
        if isinstance(X, pd.DataFrame):
            X = X.replace(self.unknown_value, dict(zip(X.columns, self.unknown_values)))
        else:
            for i, unknown_value in enumerate(self.unknown_values):
                X[X[:, i] == self.unknown_value, i] = unknown_value
        return X


def cast_to_type(X, dtype):
    if isinstance(X, pd.DataFrame):
        return X.astype(dtype)
    elif isinstance(X, np.ndarray):
        return X.astype(dtype)


def reorder_columns(X, orderly_features_names):
    orderly_features_without_dropped = deepcopy(orderly_features_names)
    for feature in orderly_features_names:
        if feature not in X.columns:
            # feature could have be transformed by one-hot encoding, so it could be present in the format
            # 'feature_name_i' where i is the index of the category
            index_of_feature = orderly_features_without_dropped.index(feature)
            orderly_features_without_dropped.remove(feature)
            # the regex ^{re.escape(feature)}_\d+$ matches exactly the format 'feature_i' where i is an integer
            # obviously it was chatgpt that gave me, so if anything does not work as expected, check this line
            one_hot_features = [col for col in X.columns if bool(re.match(rf"^{re.escape(feature)}_\d+$", col))]
            orderly_features_without_dropped[index_of_feature:index_of_feature] = one_hot_features
    return X[orderly_features_without_dropped]


def identity(X):
    return X


map_type_to_dtype = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "category": "category",
    "int32": np.int32,
    "int64": np.int64,
    "float32": np.float32,
    "float64": np.float64,
}


def create_data_preprocess_pipeline(
    categorical_features_names: Sequence[int | str],
    continuous_features_names: Sequence[int | str],
    orderly_features_names: Sequence[int | str],
    continuous_imputer: Optional[str | int | float | TransformerMixin] = "median",
    categorical_imputer: Optional[str | int | float | TransformerMixin] = "most_frequent",
    categorical_encoder: Optional[str | TransformerMixin] = "ordinal",
    handle_unknown_categories: bool = True,
    variance_threshold: Optional[float] = 0.0,
    scaler: Optional[str | TransformerMixin] = "standard",
    categorical_type: Optional[str] = "category",
    continuous_type: Optional[str] = "float32",
):
    # Continuous features
    if continuous_imputer:
        if continuous_imputer == "median":
            continuous_imputer = SimpleImputer(strategy="median")
        elif continuous_imputer == "mean":
            continuous_imputer = SimpleImputer(strategy="mean")
        elif isinstance(continuous_imputer, (int, float)):
            continuous_imputer = SimpleImputer(strategy="constant", fill_value=continuous_imputer)
        elif isinstance(continuous_imputer, TransformerMixin):
            # If a custom transformer is provided, use it directly
            continuous_imputer = continuous_imputer
        else:
            raise ValueError(f"Unknown continuous imputer: {continuous_imputer}")
    else:
        continuous_imputer = FunctionTransformer(identity)

    if variance_threshold is not None:
        variance_threshold_continuous = VarianceThreshold(threshold=variance_threshold)
    else:
        variance_threshold_continuous = FunctionTransformer(identity)

    if scaler:
        if scaler == "standard":
            scaler = StandardScaler()
        elif scaler == "minmax":
            scaler = MinMaxScaler()
        elif scaler == "robust":
            scaler = RobustScaler()
        elif isinstance(scaler, TransformerMixin):
            # If a custom scaler is provided, use it directly
            scaler = scaler
        else:
            raise ValueError(f"Unknown scaler: {scaler}")
    else:
        scaler = FunctionTransformer()

    if continuous_type:
        continuous_type = map_type_to_dtype.get(continuous_type, continuous_type)
        continuous_caster = FunctionTransformer(cast_to_type, kw_args={"dtype": continuous_type})
    else:
        continuous_caster = FunctionTransformer(identity)

    continuous_transformer = make_pipeline(continuous_imputer, variance_threshold_continuous, scaler, continuous_caster)

    # Categorical features
    if categorical_imputer:
        if categorical_imputer == "most_frequent":
            categorical_imputer = SimpleImputer(strategy="most_frequent")
        elif isinstance(categorical_imputer, (int, float)):
            categorical_imputer = SimpleImputer(strategy="constant", fill_value=categorical_imputer)
        elif isinstance(categorical_imputer, TransformerMixin):
            # If a custom transformer is provided, use it directly
            categorical_imputer = categorical_imputer
        else:
            raise ValueError(f"Unknown categorical imputer: {categorical_imputer}")
    else:
        categorical_imputer = FunctionTransformer(identity)

    if categorical_encoder:
        if categorical_encoder == "ordinal":
            if handle_unknown_categories:
                categorical_encoder = OrdinalEncoderMaxUnknownValue()
            else:
                categorical_encoder = OrdinalEncoder()
        elif categorical_encoder == "one_hot":
            categorical_encoder = OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")
        elif isinstance(categorical_encoder, TransformerMixin):
            # If a custom transformer is provided, use it directly
            categorical_encoder = categorical_encoder
        else:
            raise ValueError(f"Unknown categorical encoder: {categorical_encoder}")
    else:
        categorical_encoder = FunctionTransformer(identity)

    if variance_threshold is not None:
        variance_threshold_categorical = VarianceThreshold(threshold=variance_threshold)
    else:
        variance_threshold_categorical = FunctionTransformer(identity)

    if categorical_type:
        categorical_type = map_type_to_dtype.get(categorical_type, categorical_type)
        categorical_caster = FunctionTransformer(cast_to_type, kw_args={"dtype": categorical_type})
    else:
        categorical_caster = FunctionTransformer(identity)

    categorical_transformer = make_pipeline(
        categorical_imputer, categorical_encoder, variance_threshold_categorical, categorical_caster
    )

    # Combine continuous and categorical transformers
    transformer = ColumnTransformer(
        [
            ("continuous_transformer", continuous_transformer, continuous_features_names),
            ("categorical_transformer", categorical_transformer, categorical_features_names),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    reorder_transformer = FunctionTransformer(
        reorder_columns, kw_args={"orderly_features_names": orderly_features_names}
    ).set_output(transform="pandas")
    preprocess_pipeline = make_pipeline(transformer, reorder_transformer)
    return preprocess_pipeline


def create_target_preprocess_pipeline( 
    task: str,
    imputer: Optional[str | int | float | TransformerMixin] = None,
    categorical_encoder: Optional[str | TransformerMixin] = "ordinal",
    categorical_min_frequency: Optional[int | float] = 10,
    continuous_scaler: Optional[str | TransformerMixin] = "standard",
    categorical_type: Optional[str] = "float32",
    continuous_type: Optional[str] = "float32",
):
    if task not in ["regression", "multi_regression", "classification", "binary_classification"]:
        raise ValueError(f"Unknown task: {task}")
    if imputer:
        if isinstance(imputer, (int, float)):
            imputer = SimpleImputer(strategy="constant", fill_value=imputer)
        elif imputer == "mean":
            imputer = SimpleImputer(strategy="mean")
        elif imputer == "median":
            imputer = SimpleImputer(strategy="median")
        elif imputer == "most_frequent":
            imputer = SimpleImputer(strategy="most_frequent")
        elif isinstance(imputer, TransformerMixin):
            # If a custom imputer is provided, use it directly
            imputer = imputer
        else:
            raise ValueError(f"Unknown imputer: {imputer}")
    else:
        imputer = FunctionTransformer(identity)

    if categorical_encoder:
        if task in ("classification", "binary_classification"):
            if categorical_encoder == "ordinal":
                categorical_encoder = OrdinalEncoderMaxUnknownValue(min_frequency=categorical_min_frequency)
            elif categorical_encoder == "one_hot":
                categorical_encoder = OneHotEncoder(
                    drop="if_binary",
                    sparse_output=False,
                    handle_unknown="ignore",
                    min_frequency=categorical_min_frequency,
                )
            elif isinstance(categorical_encoder, TransformerMixin):
                # If a custom encoder is provided, use it directly
                categorical_encoder = categorical_encoder
            else:
                raise ValueError(f"Unknown encoder: {categorical_encoder}")
        else:
            categorical_encoder = FunctionTransformer(identity)
    else:
        categorical_encoder = FunctionTransformer(identity)

    if continuous_scaler:
        if task in ("regression", "multi_regression"):
            if continuous_scaler == "standard":
                continuous_scaler = StandardScaler()
            elif continuous_scaler == "minmax":
                continuous_scaler = MinMaxScaler()
            elif continuous_scaler == "robust":
                continuous_scaler = RobustScaler()
            elif isinstance(continuous_scaler, TransformerMixin):
                # If a custom scaler is provided, use it directly
                continuous_scaler = continuous_scaler
            else:
                raise ValueError(f"Unknown scaler: {continuous_scaler}")
        else:
            continuous_scaler = FunctionTransformer(identity)
    else:
        continuous_scaler = FunctionTransformer(identity)

    if task in ("classification", "binary_classification") and categorical_type:
        str_type = categorical_type
    elif task in ("regression", "multi_regression") and continuous_type:
        str_type = continuous_type
    else:
        str_type = None
    if str_type:
        dtype = map_type_to_dtype.get(str_type, str_type)
        caster = FunctionTransformer(cast_to_type, kw_args={"dtype": dtype})
    else:
        caster = FunctionTransformer(identity)

    preprocess_pipeline = make_pipeline(imputer, categorical_encoder, continuous_scaler, caster).set_output(
        transform="pandas"
    )
    return preprocess_pipeline


# def preprocess_dataset(
#     data: Optional[pd.DataFrame] = None,
#     target: Optional[pd.DataFrame] = None,
#     is_train: bool = False,
#     # if is_train, must provide at least task, categorical_features, continuous_features
#     # or provide data_preprocess_pipeline and target_preprocess_pipeline
#     task: Optional[str] = None,
#     categorical_features: Optional[Sequence[int | str]] = None,
#     continuous_features: Optional[Sequence[int | str]] = None,
#     categorical_imputer: Optional[str | int | float] = "most_frequent",
#     continuous_imputer: Optional[str | int | float] = "median",
#     categorical_encoder: Optional[str] = "ordinal",
#     handle_unknown_categories: bool = True,
#     variance_threshold: Optional[float] = 0.0,
#     data_scaler: Optional[str] = None,
#     categorical_type: Optional[type | str] = "float32",
#     continuous_type: Optional[type | str] = "float32",
#     target_imputer: Optional[str | int | float] = None,
#     categorical_target_encoder: Optional[str] = "ordinal",  # only used in classification
#     categorical_target_min_frequency: Optional[int | float] = 10,  # only used in classification
#     continuous_target_scaler: Optional[str] = None,  # only used in regression
#     categorical_target_type: Optional[type | str] = "float32",
#     continuous_target_type: Optional[type | str] = "float32",
#     *,
#     # if is_train is False, must provide the fitted data_preprocess_pipeline and target_preprocess_pipeline
#     data_preprocess_pipeline: Optional[Pipeline] = None,
#     target_preprocess_pipeline: Optional[Pipeline] = None,
# ):
#     if data is not None:
#         data.columns = data.columns.astype(str)
#     if target is not None:
#         target.columns = target.columns.astype(str)
#     if is_train:
#         if data is not None:
#             if data_preprocess_pipeline is None:
#                 data_preprocess_pipeline = create_data_preprocess_pipeline(
#                     categorical_features=categorical_features,
#                     continuous_features=continuous_features,
#                     categorical_imputer=categorical_imputer,
#                     continuous_imputer=continuous_imputer,
#                     categorical_encoder=categorical_encoder,
#                     handle_unknown_categories=handle_unknown_categories,
#                     variance_threshold=variance_threshold,
#                     scaler=data_scaler,
#                     categorical_type=categorical_type,
#                     continuous_type=continuous_type,
#                 )
#             data = data_preprocess_pipeline.fit_transform(data)
#         if target is not None:
#             if target_preprocess_pipeline is None:
#                 target_preprocess_pipeline = create_target_preprocess_pipeline(
#                     task=task,
#                     imputer=target_imputer,
#                     categorical_encoder=categorical_target_encoder,
#                     categorical_min_frequency=categorical_target_min_frequency,
#                     continuous_scaler=continuous_target_scaler,
#                     categorical_type=categorical_target_type,
#                     continuous_type=continuous_target_type,
#                 )
#             target = target_preprocess_pipeline.fit_transform(target)
#     else:
#         if data is not None:
#             if data_preprocess_pipeline is None:
#                 raise ValueError("a fitted data_preprocess_pipeline_ must be provided when is_train is False")
#             data = data_preprocess_pipeline.transform(data)
#         if target is not None:
#             if target_preprocess_pipeline is None:
#                 raise ValueError("a fitted target_preprocess_pipeline_ must be provided when is_train is False")
#             target = target_preprocess_pipeline.transform(target)
#     return data, target, data_preprocess_pipeline, target_preprocess_pipeline
