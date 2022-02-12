import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train_dataset(
    df: pd.DataFrame,
    learning_rate=0.1,
    n_estimators=200,
    n_jobs=-1,
    colsample_bytree=0.3,
    max_depth=5,
    alpha=10,
    random_state=42,
):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        learning_rate (float, optional): [description]. Defaults to 0.1.
        n_estimators (int, optional): [description]. Defaults to 200.
        n_jobs (int, optional): [description]. Defaults to -1.
        colsample_bytree (float, optional): [description]. Defaults to 0.3.
        max_depth (int, optional): [description]. Defaults to 5.
        alpha (int, optional): [description]. Defaults to 10.
        random_state (int, optional): [description]. Defaults to 42.

    Returns:
        [type]: [description]
    """

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # Create the train and test set for cross-validation of the results using
    # the train_test_split function from sklearn's model_selection module with
    #  test_size size equal to 20% of the data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    model = xgb.XGBRegressor(
        objective="reg:squarederror",  # reg:linear
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        alpha=alpha,
        random_state=random_state,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=20,
    )  # will stop after 20 round when validataion_1-rmse has not improved
    results = model.evals_result()
    return model, X_test, y_test, X_train, y_train, results


def predict_dataset(model, X_test, y_test):
    """[summary]

    Args:
        model ([type]): [description]
        X_test ([type]): [description]
        y_test ([type]): [description]

    Returns:
        [type]: [description]
    """

    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, rmse
