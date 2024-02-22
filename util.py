import os
import pandas as pd

from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from numpy import number

from sklearn import metrics
from sklearn.base import clone

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
from imblearn.metrics import geometric_mean_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate, true_positive_rate, true_negative_rate, false_negative_rate
from aif360.sklearn.metrics import equal_opportunity_difference, disparate_impact_ratio

from pathlib import Path
import sys
sys.path.append(os.path.abspath('..'))

def evaluate_model_performance(Y_target, Y_pred):

    cm = confusion_matrix(Y_target, Y_pred)
    
    print()
    print("===After training the model===")

    accuracy = accuracy_score(Y_target, Y_pred)
    precision = precision_score(Y_target, Y_pred)
    recall = recall_score(Y_target, Y_pred)
    f1 = f1_score(Y_target, Y_pred)
    f2 = fbeta_score(Y_target, Y_pred, beta=2)
    specificity = (cm[0][0] / (cm[0][0] + cm[0][1]))
    bal_accuracy = balanced_accuracy_score(Y_target, Y_pred)
    geom_mean = geometric_mean_score(Y_target, Y_pred)

    print('The accuracy score for the testing data:', accuracy)
    print('The precision score for the testing data:', precision)
    print('The recall score for the testing data:', recall)
    print('The F1 score for the testing data:', f1)
    print('The F2 score for the testing data:', f2)
    print('The specificity score for the testing data:', specificity)
    print('The balanced accuracy score for the testing data:', bal_accuracy)
    print('The G mean score for the testing data:', geom_mean)

    #Ploting the confusion matrix
    print(cm)

    return [accuracy, precision, recall, f1, f2, specificity, bal_accuracy, geom_mean]

def evaluate_model_fairness(Y_target, Y_pred, sensitive_features):

    #fairness metrics to be used
    metrics_dict = {
        "true_positive_rate" : true_positive_rate,
        "balanced_accuracy" : balanced_accuracy_score,
        "selection_rate" : selection_rate,
        # "false_negative_rate" : false_negative_rate,        
    }

    metric_frame = MetricFrame(
        metrics=metrics_dict,
        y_true=Y_target,
        y_pred=Y_pred,
        sensitive_features=sensitive_features
    )

    # The disaggregated metrics
    print(metric_frame.by_group)

    metric_frame.by_group.plot.bar(
        subplots=True, layout=[1, 4], figsize=[16, 4], legend=None, rot=-45, position=1
    )

    dem_par_diff = demographic_parity_difference(
        y_true=Y_target,
        y_pred=Y_pred,
        sensitive_features=sensitive_features
    )

    eq_odd_diff = equalized_odds_difference(
            y_true=Y_target,
            y_pred=Y_pred,
            sensitive_features=sensitive_features
    )

    eq_opp_diff = equal_opportunity_difference(
            y_true=Y_target,
            y_pred=Y_pred,
            prot_attr=sensitive_features
    )

    disparate_impact_rat = disparate_impact_ratio(
        y_true=Y_target,
        y_pred=Y_pred,
        prot_attr=sensitive_features,
        priv_group='Caucasian',
        pos_label=True
    )

    print('The Demographic parity difference score on the trained model:', dem_par_diff)
    print('The Equalized odds difference score on the trained model:', eq_odd_diff)
    print('The Equal opportunity difference score on the trained model:', eq_opp_diff)
    print('The Disparate impact ratio on the trained model:', disparate_impact_rat)

    return [dem_par_diff, eq_odd_diff, eq_opp_diff, disparate_impact_rat, metric_frame]
    
def evaluate_train_data_fairness(X_train, Y_train, A_train):
        
    #fairness metrics to be used
    metrics_dict = {
        "balanced_accuracy" : balanced_accuracy_score,
        "true_positive_rate" : true_positive_rate,
        "selection_rate" : selection_rate,
    }

    tree_mf = MetricFrame(
        metrics=metrics_dict,
        y_true=Y_train,
        y_pred=Y_train,
        sensitive_features=A_train
    )

    # The disaggregated metrics
    print()
    print("===Before training the model===")
    print(tree_mf.by_group)

    dem_par_diff = demographic_parity_difference(
        y_true=Y_train,
        y_pred=Y_train,
        sensitive_features=A_train
    )

    disparate_impact_rat = disparate_impact_ratio(
        y_true=Y_train,
        prot_attr=A_train,
        priv_group='Caucasian',
        pos_label=True
    )

    print('The Demographic parity difference score on the training data:', dem_par_diff)
    print("The Disparate impact ratio on the training data:", disparate_impact_rat)

    return [dem_par_diff, disparate_impact_rat]

def train_model_on_datasets(model, type="scikit"):

    # load datasets
    data_path = Path(os.getcwd()).parent.parent / "data" / "dataset_diabetes" / "clsf_data"

    target_variable = "readmit_30_days"
    sensitive_attribute = "race"

    X_test = pd.read_csv(data_path / "X_test_split.csv")
    X_A_test = pd.read_csv(data_path / "X_A_test_split.csv")
    Y_test = pd.read_csv(data_path / "Y_test_split.csv")[target_variable]
    A_test = pd.read_csv(data_path / "A_test_split.csv")[sensitive_attribute]

    X_train = pd.read_csv(data_path / "X_train_split.csv")
    X_A_train = pd.read_csv(data_path / "X_A_train_split.csv")
    Y_train = pd.read_csv(data_path / "Y_train_split.csv")[target_variable]
    A_train = pd.read_csv(data_path / "A_train_split.csv")[sensitive_attribute]

    X_train_res_target_wos = pd.read_csv(data_path / "X_train_res_target_wos.csv")
    Y_train_res_target_wos = pd.read_csv(data_path / "Y_train_res_target_wos.csv")[target_variable]
    A_train_res_target_wos = pd.read_csv(data_path / "A_train_res_target_wos.csv").iloc[:, 0]

    X_A_train_res_target_ws = pd.read_csv(data_path / "X_A_train_res_target_ws.csv")
    Y_train_res_target_ws = pd.read_csv(data_path / "Y_train_res_target_ws.csv")[target_variable]
    A_train_res_target_ws = pd.read_csv(data_path / "A_train_res_target_ws.csv").iloc[:, 0]

    X_train_res_sensitive_wos = pd.read_csv(data_path / "X_train_res_sensitive_wos.csv")
    Y_train_res_sensitive_wos = pd.read_csv(data_path / "Y_train_res_sensitive_wos.csv")[target_variable]
    A_train_res_sensitive_wos = pd.read_csv(data_path / "A_train_res_sensitive_wos.csv").iloc[:, 0]

    X_A_train_res_sensitive_ws = pd.read_csv(data_path / "X_A_train_res_sensitive_ws.csv")
    Y_train_res_sensitive_ws = pd.read_csv(data_path / "Y_train_res_sensitive_ws.csv")[target_variable]
    A_train_res_sensitive_ws = pd.read_csv(data_path / "A_train_res_sensitive_ws.csv").iloc[:, 0]

    X_train_res_multiv_wos = pd.read_csv(data_path / "X_train_res_multiv_wos.csv")
    Y_train_res_multiv_wos = pd.read_csv(data_path / "Y_train_res_multiv_wos.csv")[target_variable]
    A_train_res_multiv_wos = pd.read_csv(data_path / "A_train_res_multiv_wos.csv").iloc[:, 0]

    X_A_train_res_multiv_ws = pd.read_csv(data_path / "X_A_train_res_multiv_ws.csv")
    Y_train_res_multiv_ws = pd.read_csv(data_path / "Y_train_res_multiv_ws.csv")[target_variable]
    A_train_res_multiv_ws = pd.read_csv(data_path / "A_train_res_multiv_ws.csv").iloc[:, 0]

    X_train_list = [X_train, X_A_train, X_train_res_target_wos, X_A_train_res_target_ws, X_train_res_sensitive_wos, \
              X_A_train_res_sensitive_ws, X_train_res_multiv_wos, X_A_train_res_multiv_ws]

    Y_train_list = [Y_train, Y_train, Y_train_res_target_wos, Y_train_res_target_ws, Y_train_res_sensitive_wos, \
                    Y_train_res_sensitive_ws, Y_train_res_multiv_wos, Y_train_res_multiv_ws]

    A_train_list = [A_train, A_train, A_train_res_target_wos, A_train_res_target_ws, A_train_res_sensitive_wos, \
                    A_train_res_sensitive_ws, A_train_res_multiv_wos, A_train_res_multiv_ws]

    X_test_list = [X_test, X_A_test, X_test, X_A_test, X_test, X_A_test, X_test, X_A_test]

    dataset_ref = [ 'no resampling, w/o sensitive attribute', 'no resampling, with sensitive attribute', \
                    'target-resample, w/o sensitive attribute', 'target-resample, with sensitive attribute', \
                    'sensitive-resample, w/o sensitive attribute', 'sensitive-resample, with sensitive attribute', \
                    'multivariate-resample, w/o sensitive attribute', 'multivariate-resample, with sensitive attribute']
    
    train_data_fairness_results = []
    performance_results = []
    fairness_results = []
    
    for (X_train_i, Y_train_i, A_train_i, X_test_i, ref_i) in zip(X_train_list, Y_train_list, A_train_list, X_test_list, dataset_ref):
        print()
        print(ref_i)
        model_copy = clone(model)
        model_pred = None
        if type=='scikit':
            model_copy.fit(X_train_i, Y_train_i)
            model_pred = model_copy.predict(X_test_i)
            print(metrics.RocCurveDisplay.from_estimator(model_copy, X_test_i, Y_test))
        elif type=='adversarial':
            ct = make_column_transformer(
                (
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("normalizer", StandardScaler()),
                        ]
                    ),
                    make_column_selector(dtype_include=number),
                ),
                (
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(drop="if_binary", sparse=False)),
                        ]
                    ),
                    make_column_selector(dtype_include="category"),
                ),
            )

            X_prep_train_i = ct.fit_transform(X_train_i) # Only fit on training data!
            X_prep_test_i = ct.transform(X_test_i)
            model_copy.fit(X_prep_train_i, Y_train_i, sensitive_features=A_train_i)
            model_pred = model_copy.predict(X_prep_test_i)
        elif type=='postproc':
            model_copy.fit(X_train_i, Y_train_i, sensitive_features=A_train_i)
            model_pred = model_copy.predict(X_test_i, sensitive_features=A_test)
        
        # Predicting on the test data        
        train_data_fairness_results_i = evaluate_train_data_fairness(X_train_i, Y_train_i, A_train_i)
        performance_results_i = evaluate_model_performance(Y_test, model_pred)
        fairness_results_i = evaluate_model_fairness(Y_test, model_pred, A_test)

        train_data_fairness_results.append(train_data_fairness_results_i)
        performance_results.append(performance_results_i)
        fairness_results.append(fairness_results_i)

    return train_data_fairness_results, performance_results, fairness_results