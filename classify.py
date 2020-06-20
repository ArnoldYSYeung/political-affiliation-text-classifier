"""
Name: Arnold Yeung
Date: February 11 2019
Description: Conduct classification tests with multiple classifiers
"""

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn import svm, metrics, ensemble, neural_network
from scipy import stats
import numpy as np
import argparse
import sys
import os
import datetime
import csv
import warnings
warnings.filterwarnings("ignore")

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return float(sum(C.diagonal())) / C.sum()


def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recall = []
    label = 0
    for row in C:     #   for each true label
        recall.append(float(row[label]) / row.sum())
        label += 1
    return recall  

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precision = []
    label = 0
    for col in C.T: #   for each predicted label
        precision.append(float(col[label]) / col.sum())
        label += 1
    return precision

def classify(classifier, X_train, X_test, y_train, y_test):
    
    if classifier == 'lin_svm':
        clf = svm.SVC(kernel='linear', max_iter=1000)
        clf_tag = 1
    elif classifier == 'radial_svm':
        clf = svm.SVC(kernel='rbf', max_iter=1000, gamma=2)
        clf_tag = 2
    elif classifier == 'random_forest':
        clf = ensemble.RandomForestClassifier(n_estimators=10, max_depth=5)
        clf_tag = 3
    elif classifier == 'mlp':
        clf = neural_network.MLPClassifier(alpha=0.05)
        clf_tag = 4
    elif classifier == 'adaboost':
        clf = ensemble.AdaBoostClassifier()
        clf_tag = 5
    else:
        print('Classifier not found.  Using Linear SVM...')
        clf = svm.SVC(kernel='linear')
        clf_tag = 0
    
    print(datetime.datetime.now())
    print("Fitting classifier...")
    clf.fit(X_train, y_train)
    print("Predicting test data...")
    y_predict = clf.predict(X_test)
    
    #   confusion matrix where row = true label, col = predicted label
    C = metrics.confusion_matrix(y_test, y_predict, labels=[0, 1, 2, 3])
    print(C)
    print(datetime.datetime.now())
    
    return C, y_predict, clf_tag

def get_accuracies(C):
    #   calculate accuracies
    acc = accuracy(C)
    rec = recall(C)
    prec = precision(C)
    return acc, rec, prec

def get_report_line(tag, acc=None, rec=None, prec=None, C=None):
    #   add to report
    if acc is not None:
        line = [str(tag), acc]
    else:
        line = [str(tag)]
    if rec is not None:
        for value in rec:
            line.append(value) 
    if prec is not None:
        for value in prec:
            line.append(value)
    if type(C) != type(None):
        for row in C:
            for value in row:
                line.append(value)
    return line

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    
    report_file = 'a1_3.1.csv'
    test_data = False
    safe_mode = False
    iBest = 0
    bestAcc = 0
    clfs = ['lin_svm', 'radial_svm', 'random_forest', 'mlp', 'adaboost']
    
    npz_data = np.load(filename)
    data = npz_data['arr_0']
    
    #   split data into train and test sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(data[:,:173], data[:, 173],
                                                        train_size=0.8) 
    
    if test_data is True:
        print("WARNING: Using test method...")
        X_train = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        y_train = np.array([1, 1, 2, 2])
        X_test = np.array([[-0.8, -1], [3, 0]])
        y_test = [1, 3]
    
    report = []
    
    #   run all classifiers
    for clf in clfs:
        print("Running " + clf + " classifier...")
        C, y_predict, clf_tag = classify(clf, X_train, X_test, y_train, y_test)
        acc, rec, prec = get_accuracies(C)
        print("Acc: " + str(acc) + "  Rec: " + str(rec) + "  Prec: " + str(prec))
        line = get_report_line(clf_tag, acc, rec, prec, C)
        report.append(line)
        
        #   save data
        if safe_mode is True:
            save_file = 'class31_' + clf + '.csv'
            with open(save_file, 'w') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(line)
                wr.writerow(y_test)
                wr.writerow(y_predict)
        
        #   update best accuracy
        if acc > bestAcc:
            print("Best accuracy updated.")
            iBest = clf_tag
            bestAcc = acc    
    
    print("Classifier " + str(iBest) + " has the best accuracy: " 
          + str(bestAcc))
    
    #   write everything to the report file
    with open(report_file, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(report)
    
    csvFile.close()

    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  
    
    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
       '''
    safe_mode = False
    report_file = 'a1_3.2.csv'
       
    clf_tag_dict = {1: 'lin_svm',
                    2: 'radial_svm',
                    3: 'random_forest',
                    4: 'mlp',
                    5: 'adaboost'}
    report = []
   
    #    number of training samples per trial
    training_samples_per_trial = [1000, 5000, 10000, 15000, 20000]
    
    line = []
    
    for trial in training_samples_per_trial:
        print("Running " + clf_tag_dict[iBest] + " classifier trial for " + 
                  str(trial) + " samples...")
        #    change number of training samples
        X_train_trial, _, y_train_trial, _ = train_test_split(X_train, y_train, 
                                                         train_size=trial)
       
        C, y_predict, clf_tag = classify(clf_tag_dict[iBest], X_train_trial, X_test, 
                                y_train_trial, y_test)
       
        acc, rec, prec = get_accuracies(C)
        print("Acc: " + str(acc) + "  Rec: " + str(rec) + "  Prec: " + str(prec))
        line.append(acc)
    
        #   save data
        if safe_mode is True:
            save_file = 'class32_' + clf_tag_dict[iBest] + '_' + str(trial) + '.csv'
            with open(save_file, 'w') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(line)
    
        #   save training data if number is 1000
        if trial == 1000:
            X_1k = X_train_trial
            y_1k = y_train_trial
    
    report.append(line)
    
    #   write everything to the report file
    with open(report_file, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(report)
    
    csvFile.close()
    
    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    safe_mode = True
    report_file = 'a1_3.3.csv'
    
    report = []
    num_best_feats = [5, 10, 20, 30, 40, 50]
    clf_tag_dict = {1: 'lin_svm',
                    2: 'radial_svm',
                    3: 'random_forest',
                    4: 'mlp',
                    5: 'adaboost'}
    
    print('Calculating p-values for different numbers of top features on 1k training dataset...')
    save_p_file = 'a1_3.3_pvalue_1k.csv'
    save_report = []
    for num_feats in num_best_feats:
        selector = SelectKBest(f_classif, k=num_feats)
        X_new = selector.fit_transform(X_1k, y_1k)
        #   calculate p-value for features 
        pp = selector.pvalues_
        #   add p-values of top features to report
        top_feats = selector.get_support(indices=True)
        pp = pp[top_feats]
        line = pp.tolist()
        line.insert(0, num_feats)
        #   save top features and p-values
        if safe_mode is True:
            save_report.append(top_feats)
            save_report.append(line[1:])
    #   write top features and p-values to file
    if safe_mode is True:
        with open(save_p_file, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(save_report)
        csvFile.close()
        
        
    print('Calculating p-values for different numbers of top features on 32k training dataset...')
    save_p_file = 'a1_3.3_pvalue_32k.csv'
    save_report = []
    for num_feats in num_best_feats:
        selector = SelectKBest(f_classif, k=num_feats)
        X_new = selector.fit_transform(X_train, y_train)
        #   calculate p-value for features 
        pp = selector.pvalues_
        #   add p-values of top features to report
        top_feats = selector.get_support(indices=True)  
        pp = pp[top_feats]
        line = pp.tolist()
        line.insert(0, num_feats)
        report.append(line)
        #   save top features and p-values
        if safe_mode is True:
            save_report.append(top_feats)
            save_report.append(line[1:])
    #   write top features and p-values to file
    if safe_mode is True:
        with open(save_p_file, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(save_report)
        csvFile.close()
    
    line = []
    
    #   find top 5 features for 1k dataset
    print("Selecting top 5 features for 1k training dataset...")
    selector = SelectKBest(f_classif, k=5)
    X_new = selector.fit_transform(X_1k, y_1k)
    top_feats = selector.get_support(indices=True)  
    print("The top features are : " + str(top_feats))
    #   classify using selected features
    C, y_predict, clf_tag = classify(clf_tag_dict[i], X_new, X_test[:, top_feats], 
                                     y_1k, y_test)
    acc, rec, prec = get_accuracies(C)
    print("Accuracy for " + clf_tag_dict[i] + " classifier is " + str(acc) 
                        + " for 1k training dataset with top 5 features.")
    line.append(acc)
    
    #   find top 5 features for 32k dataset
    print("Selecting top 5 features for 32k training dataset...")
    selector = SelectKBest(f_classif, k=5)
    X_new = selector.fit_transform(X_train, y_train)
    top_feats = selector.get_support(indices=True)    
    print("The top features are : " + str(top_feats))
    #   classify using selected features
    C, y_predict, clf_tag = classify(clf_tag_dict[i], X_new, X_test[:, top_feats], 
                                     y_train, y_test)
    acc, rec, prec = get_accuracies(C)
    print("Accuracy for " + clf_tag_dict[i] + " classifier is " + str(acc) 
                        + " for 32k training dataset with top 5 features.")
    line.append(acc)
    
    report.append(line)
    
    #   write everything to the report file
    with open(report_file, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(report)
    csvFile.close()
        
    
def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
        
    report_file = 'a1_3.4.csv'
    
    report = []
    clf_tag_dict = {1: 'lin_svm',
                    2: 'radial_svm',
                    3: 'random_forest',
                    4: 'mlp',
                    5: 'adaboost'}
    
    #   create dictionary to store classification accuracies per fold
    clf_accuracies = {}
    for key in clf_tag_dict.keys():
        clf_accuracies[key] = []
        
    
    npz_data = np.load(filename)
    data = npz_data['arr_0']
    #   split data to features and labels
    X = data[:,:173]
    y = data[:, 173]
    
    k_folds = KFold(n_splits=5, shuffle=True)
    
    fold_num = 0
    for train_idx, test_idx in k_folds.split(X):    #   for each fold
        line = []
        print("Cross-validating for fold " + str(fold_num))
        fold_num += 1
        print("Training with " + str(len(train_idx)) + "; testing with " +
              str(len(test_idx)) + "...");
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        #   classify using all classifiers for each fold
        for clf_idx in clf_tag_dict.keys():
            C, y_predict, _ = classify(clf_tag_dict[clf_idx], X_train, X_test, 
                                     y_train, y_test)
            acc, rec, prec = get_accuracies(C)
            clf_accuracies[clf_idx].append(acc)       #   append the classification accuracy
            print("Accuracy for " + clf_tag_dict[clf_idx] + " classifier is " 
                  + str(acc) + ".")
            line.append(acc)
        report.append(line)
    
    print(clf_accuracies)

    #   find the classifier with max accuracy
    mean_clf_accuracies = [float(sum(clf_accuracies[key]))/len(clf_accuracies[key]) 
                            for key in clf_accuracies.keys()]
    max_clf = mean_clf_accuracies.index(max(mean_clf_accuracies)) + 1       # plus 1 because index
    print("The best classifier is " + clf_tag_dict[max_clf] + " with mean accuracy of "
          + str(mean_clf_accuracies[max_clf-1]) + ".")
    
    #   calculate p-value between best classifier and all other classifiers
    line = []
    for clf in [key for key in clf_accuracies.keys() if key != max_clf]:
        t_score, p_value = stats.ttest_rel(clf_accuracies[max_clf], clf_accuracies[clf])
        line.append(p_value)
        print("The p-value for " + clf_tag_dict[max_clf] + " and " + clf_tag_dict[clf] +
              " is " + str(p_value) + ".")
    report.append(line)
    
    #   write everything to the report file
    with open(report_file, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(report)
    csvFile.close()
        
    

def main( args ):
    
    processes = [1, 2, 3, 4]
    if 1 in processes:
        X_train, X_test, y_train, y_test, iBest = class31(args['input'])
    else:
        filename = args.input
        npz_data = np.load(filename)
        data = npz_data['arr_0']
        X_train, X_test, y_train, y_test = train_test_split(data[:,:173], 
                                                            data[:, 173],
                                                            train_size=0.8) 
        iBest = 3
    
    if 2 in processes:
        X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    else:
        X_1k, _, y_1k, _ = train_test_split(X_train, y_train, 
                                                         train_size=1000)
    
    if 3 in processes:
        class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    
    if 4 in processes:
        filename = args['input']
        class34(filename, iBest)
    

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()
    """
    
    args = {'input':        'feats.npz'}
    
    # TODO : complete each classification experiment, in sequence.
    main( args )