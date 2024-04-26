import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data
    # X_train_normalized = X_train / 255.0
    # X_test_normalized = X_test / 255.0
    
    X_train_normalized = 2 * ((X_train - np.mean(X_train))/(np.max(X_train) - np.min(X_train))) -1
    X_test_normalized = 2 * ((X_test- np.mean(X_test))/(np.max(X_test) - np.min(X_test))) -1

    return X_train_normalized, X_test_normalized


def plot_metrics(metrics) -> None:
    # unpack the data from the metrics list
    ks, accuracies, precisions, recalls, f1_scores = zip(*metrics)
    # ks, accuracies = zip(*metrics)


    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,5) ,sharex=True, sharey=True)

    axs[0][0].plot(ks, accuracies, color='y' )
    axs[0][0].set_xlabel('ks')
    axs[0][0].set_ylabel('accuracies')
    axs[0][0].set_title('accuracies vs. k')

    # plt.show()


    axs[0][1].plot(ks, precisions,color='b')
    axs[0][1].set_xlabel('ks')
    axs[0][1].set_ylabel('precisions')
    axs[0][1].set_title('precisions vs. k')

    # plt.show()

    axs[1][0].plot(ks, recalls,color='o'  )
    axs[1][0].set_xlabel('ks')
    axs[1][0].set_ylabel('recalls')
    axs[1][0].set_title('recalls vs. k')

    # plt.show()

    axs[1][1].plot(ks, f1_scores,color='r'  )
    axs[1][1].set_xlabel('ks')
    axs[1][1].set_ylabel('f1_scores')
    axs[1][1].set_title('f1_scores vs. k')

    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    plt.show()

    # plot the accuracy
    plt.plot(ks, accuracies, label='Accuracy')
    

    # plot the precision
    plt.plot(ks, precisions, label='Precision')
    

    # # plot the recall
    plt.plot(ks, recalls, label='Recall')
    

    # # plot the f1 score
    plt.plot(ks, f1_scores, label='F1 Score')
    

    # add labels and title to the plot
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.title('Metrics vs. k')

    # add legend to the plot
    plt.legend()

    # show the plot
    plt.show()