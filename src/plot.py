import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


def cross_validation_report(X_train, y_train):
    """[summary]

    Args:
        X_train ([type]): [description]
        y_train ([type]): [description]
    """
    DM_train = xgb.DMatrix(data=X_train, label=y_train)
    params = {
        "objective": "reg:squarederror",
        "colsample_bytree": 0.3,
        "learning_rate": 0.1,
        "max_depth": 5,
        "alpha": 10,
    }
    cv_results = xgb.cv(
        nfold=3,
        dtrain=DM_train,
        params=params,
        num_boost_round=50,
        early_stopping_rounds=10,
        metrics="rmse",
        as_pandas=True,
        seed=123,
    )

    cv_results.to_csv("cross_validation_report.csv", index=False)
    return params


def xgboost_tree_importance_plots(df: pd.DataFrame, params: dict):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        params (dict): [description]
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        # Convert the dataset into an optimized data structure called Dmatrix
        # that XGBoost supports and gives it acclaimed performance and efficiency gains.
        data_dmatrix = xgb.DMatrix(data=X, label=y)
        xgb_reg_1 = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

        # plt.figure(figsize=(16, 9))
        xgb.plot_tree(xgb_reg_1, num_trees=0)
        fig = plt.gcf()
        fig.set_size_inches(150, 100)
        fig.savefig("xgboost_tree.png")

        plt.figure(figsize=(16, 16))
        xgb.plot_importance(xgb_reg_1)
        plt.rcParams["figure.figsize"] = [50, 50]
        plt.savefig("importance_plot.png")


def optimal_num_tree_graph(model, results):
    """[summary]

    Args:
        model ([type]): [description]
        results ([type]): [description]
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        plt.figure(figsize=(10, 7))
        plt.plot(results["validation_0"]["rmse"], label="Training loss")
        plt.plot(results["validation_1"]["rmse"], label="Validation loss")
        plt.axvline(
            x=model.best_ntree_limit,
            ymin=0,
            ymax=14,
            color="gray",
            label="Optimal tree number",
        )
        plt.xlabel("Number of Tree")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig("optimal_num_tree.png")


def actual_vs_predicted_graph(y_test, y_pred):
    """[summary]

    Args:
        y_test ([type]): [description]
        y_pred ([type]): [description]
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        plt.figure(figsize=(16, 9))
        plt.plot(np.arange(0, len(y_test)), y_test, label="Actual value")
        plt.plot(np.arange(0, len(y_test)), y_pred, label="Predicted value")
        plt.xlabel("samples")
        plt.ylabel("Price")
        plt.legend()
        plt.title("Actual and predicted values")
        plt.savefig("actual_vs_predicted.png")
