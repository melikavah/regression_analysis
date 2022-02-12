import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def boston_info(description: str):
    """Writing the description string into a text file.

    Args:
        description (str): [description]
    """
    text_file = open("boston_housing_decsription.txt", "w")
    text_file.write(description)


def pair_dist(df: pd.DataFrame):
    fig = sns.pairplot(df)
    fig.savefig("distribution.png")


def heatmap_corr_plot(df: pd.DataFrame):
    correlation_matrix = df.corr().round(2)
    plt.figure(figsize=(16, 9))
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.savefig("correlation.png")
