import pprint
import warnings

import pandas as pd
from sklearn.datasets import load_boston

from .eda import boston_info, heatmap_corr_plot, pair_dist
from .plot import (
    actual_vs_predicted_graph,
    cross_validation_report,
    optimal_num_tree_graph,
    xgboost_tree_importance_plots,
)
from .train import predict_dataset, train_dataset

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    boston_dataset = load_boston()

# data: contains the information for various houses
# target: prices of the house
# feature_names: names of the features
# DESCR: describes the dataset
boston_df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston_df["MEDV"] = boston_dataset.target

##############################################################
############## EDA (Exploratory data analysis) ################
# extracting the dataset description in a txt file.
boston_info(boston_dataset.DESCR)
pair_dist(boston_df)
heatmap_corr_plot(boston_df)
pprint.pprint(boston_df.describe())
############################################################
############# Training #####################################
model, X_test, y_test, X_train, y_train, results = train_dataset(boston_df)
y_pred, rmse = predict_dataset(model, X_test, y_test)
print(f"The root mean squared error (RMSE) is {rmse}")
params = cross_validation_report(X_train, y_train)
xgboost_tree_importance_plots(boston_df, params)
optimal_num_tree_graph(model, results)
actual_vs_predicted_graph(y_test, y_pred)
