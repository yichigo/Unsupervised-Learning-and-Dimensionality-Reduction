import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import sklearn.tree as tree
from sklearn.metrics import plot_confusion_matrix

# Classifiers 
from sklearn.neural_network import MLPClassifier
from plots import plot_roc

import os
import sys

dir_in = '../data/'
dir_out = '../plots/'


def main(argv):
    DR = 'None'
    Cluster = 'None'
    onehot_vars = [] # original variables has finished one-hot in preprocessing
    if len(argv) > 1:
        DR = argv[1] # dimension reduction: PCA, ICA, RP, 
    if len(argv) > 2:
        Cluster = argv[2] # Cluster algorithm: KMeans, EM
        onehot_vars = ['Cluster']

    print(DR)
    print(Cluster)

    # read data
    fn_in = 'heart_'+ DR + '_' + Cluster +'.csv'
    df = pd.read_csv(dir_in + fn_in)

    if len(onehot_vars)>0:
        # one-hot encoding: Cluster to Cluster_1, Cluster_2, ...( drop Cluater_0)
        for var in onehot_vars:
            df_add = pd.get_dummies(df[[var]].astype(str),prefix=[var], drop_first=True)
            df = pd.concat([df, df_add], axis=1)

        df.drop(onehot_vars, axis=1, inplace=True)

        # move target to the last column
        df_target = df.pop('target') 
        df['target'] = df_target

    # split dataset
    RANDOM_STATE_DATA = 0
    X = df.drop(['target'],axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE_DATA) 

    # train NN model
    RANDOM_STATE_MODEL = 42
    clf = MLPClassifier(random_state = RANDOM_STATE_MODEL, 
                        hidden_layer_sizes=30,#CV_clf.best_params_['hidden_layer_sizes']
                        activation = 'identity',#CV_clf.best_params_['activation'], 
                        solver='adam', #CV_clf.best_params_['solver'], #
                        max_iter = 1000#CV_clf.best_params_['max_iter'] #
                       )
    clf.fit(X_train, y_train)

    # plot Confusion Matrix
    labels = [0,1]
    label_names = ['0','1']
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams.update({'font.size': 15})

    plot_confusion_matrix(clf, X_train, y_train,labels = labels, display_labels = label_names, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Train):' + '\nDimension Reduction: '+ DR + '\nCluster: ' + Cluster, fontsize = 15)
    plt.tight_layout()
    plt.savefig(dir_out + DR + '_' + Cluster +'_CM_train.png')
    plt.close()

    plot_confusion_matrix(clf, X_test, y_test,labels = labels, display_labels = label_names, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Test):' + '\nDimension Reduction: '+ DR + '\nCluster: ' + Cluster, fontsize = 15)
    plt.tight_layout()
    plt.savefig(dir_out + DR + '_' + Cluster +'_CM_test.png')
    plt.close()


    # plot ROC 
    y_train_pred = clf.predict_proba(X_train)[:,1]
    y_test_pred = clf.predict_proba(X_test)[:,1]

    label_names = ['ROC Curve (Train):' + '\nDimension Reduction: '+ DR + '\nCluster: ' + Cluster]
    length = len(y_train)
    plot_roc(y_train.values.reshape(length,1), y_train_pred.reshape(length,1), label_names)
    plt.savefig(dir_out + DR + '_' + Cluster + '_ROC_train.png')
    plt.close()

    label_names = ['ROC Curve (Test):' + '\nDimension Reduction: '+ DR + '\nCluster: ' + Cluster]
    length = len(y_test)
    plot_roc(y_test.values.reshape(length,1), y_test_pred.reshape(length,1), label_names)
    plt.savefig(dir_out + DR + '_' + Cluster + '_ROC_test.png')
    plt.close()


if __name__ == '__main__':
    main(sys.argv)

