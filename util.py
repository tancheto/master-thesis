from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
from imblearn.metrics import geometric_mean_score

from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate
from aif360.sklearn.metrics import equal_opportunity_difference

def evaluate_model_performance(Y_target, Y_pred):

    cm = confusion_matrix(Y_target, Y_pred)

    print('The accuracy score for the testing data:', accuracy_score(Y_target, Y_pred))
    print('The precision score for the testing data:', precision_score(Y_target, Y_pred))
    print('The recall score for the testing data:', recall_score(Y_target, Y_pred))
    print('The F1 score for the testing data:', f1_score(Y_target, Y_pred))
    print('The F2 score for the testing data:', fbeta_score(Y_target, Y_pred, beta=2))
    print('The specificity score for the testing data:', (cm[0][0] / (cm[0][0] + cm[0][1])))
    print('The balanced accuracy score for the testing data:', balanced_accuracy_score(Y_target, Y_pred))
    print('The G mean score for the testing data:', geometric_mean_score(Y_target, Y_pred))

    #Ploting the confusion matrix
    print(cm)

def evaluate_model_fairness(Y_target, Y_pred, sensitive_features):

    #fairness metrics to be used
    metrics_dict = {
        "selection_rate": selection_rate
    }

    tree_mf = MetricFrame(
        metrics=metrics_dict,
        y_true=Y_target,
        y_pred=Y_pred,
        sensitive_features=sensitive_features
    )

    # The disaggregated metrics
    tree_mf.by_group

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

    print('The Demographic parity difference score for the testing data:', dem_par_diff)
    print('The Equalized odds difference score for the testing data:', eq_odd_diff)
    print('The Equal opportunity difference score for the testing data:', eq_opp_diff)