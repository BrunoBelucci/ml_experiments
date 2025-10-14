from __future__ import annotations
import re
from copy import deepcopy
from typing import Optional, Sequence
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (FunctionTransformer, OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler,
                                    RobustScaler)
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split


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
            possible_longer_features = [feat for feat in orderly_features_names if feat.startswith(f"{feature}_") and len(feat) > len(feature)]
            # feature could have be transformed by one-hot encoding, so it could be present in the format
            # feature + '_' + str(category)
            index_of_feature = orderly_features_without_dropped.index(feature)
            orderly_features_without_dropped.remove(feature)
            one_hot_features = []
            for col in X.columns:
                if col.startswith(f"{feature}_"):
                    col_comes_from_longer_feature = any(col.startswith(f"{longer_feature}_") for longer_feature in possible_longer_features)
                    if not col_comes_from_longer_feature:
                        one_hot_features.append(col)
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

    preprocess_pipeline = make_pipeline(imputer, categorical_encoder, continuous_scaler, caster)
    preprocess_pipeline.set_output(transform="pandas")
    return preprocess_pipeline


def train_test_split_forced(train_data, train_target, test_size_pct, random_state=None, stratify=None):
    if train_target is None:
        X_train, X_valid = train_test_split(
            train_data, test_size=test_size_pct, random_state=random_state, stratify=stratify
        )
        y_train = None
        y_valid = None
    else:
        if stratify is not None:
            number_of_classes = max(train_target.nunique())
            # If we sample less than the number of classes, we cannot have at lest one example per class in the split
            # For example, we cannot sample 10 examples if we have 11 classes, because we will have at least one
            # class with 0 examples in the validation set.
            # NOTE: apparently this does not ensure that we have at least one example per class in the validation set,
            # but it is the best we can do for now...
            if test_size_pct * len(train_data) < number_of_classes:
                test_size_pct = max(train_target.nunique())
        try:
            X_train, X_valid, y_train, y_valid = train_test_split(
                train_data, train_target, test_size=test_size_pct, random_state=random_state, stratify=stratify
            )
        except ValueError as exception:
            warnings.warn(
                f"Got {exception} when splitting the data, trying to fix it by artificially increasing "
                f"the number of examples of the least frequent class."
            )
            # Probably the error is:
            # The least populated class in y has only 1 member, which is too few. The minimum number of groups
            # for any class cannot be less than 2.
            class_counts = train_target.value_counts()
            only_1_member_classes = class_counts[class_counts == 1]
            for only_1_member_class in only_1_member_classes.index:
                if isinstance(only_1_member_class, tuple):
                    only_1_member_class = only_1_member_class[0]
                index_of_only_1_member_class = train_target[train_target.iloc[:, 0] == only_1_member_class].index[0]
                train_data = pd.concat([train_data, train_data.loc[[index_of_only_1_member_class]]]).reset_index(drop=True)
                train_target = pd.concat([train_target, train_target.loc[[index_of_only_1_member_class]]]).reset_index(
                    drop=True
                )
            if stratify is not None:
                stratify = train_target
            X_train, X_valid, y_train, y_valid = train_test_split(
                train_data, train_target, test_size=test_size_pct, random_state=random_state, stratify=stratify
            )
    return X_train, X_valid, y_train, y_valid
