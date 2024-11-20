import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


# binary classification metrics
def metrics_for_binary_classification(type_1, type_2, predict, test_label, score_for_AUC=0):

    predict = np.array(predict)
    test_label = np.array(test_label)
    score_for_AUC = np.array(score_for_AUC)

    correct = 0
    total = len(test_label)
    TP, TN, FN, FP = 0, 0, 0, 0

    for i in range(total):
        if predict[i] == test_label[i]:
            correct += 1

        if test_label[i] == type_1:
            if predict[i] == type_1:
                TP += 1
            elif predict[i] == type_2:
                FN += 1
        elif test_label[i] == type_2:
            if predict[i] == type_2:
                TN += 1
            elif predict[i] == type_1:
                FP += 1

    Accuracy = correct / total
    if TP + FN == 0:
        Sensitivity = 0
    elif TP + FN != 0:
        Sensitivity = TP / (TP + FN)
    if TN + FP == 0:
        Specificity = 0
    elif TN + FP!=0:
        Specificity = TN / (TN + FP)
    if score_for_AUC.any() != 0:
        AUC = roc_auc_score(test_label, score_for_AUC)
    else:
        AUC = 0
        print('AUC is not calculated, default = 0')
    if TP + FP==0:
        Precision = 0
    elif TP + FP!=0:
        Precision = TP /(TP + FP)
    Balanced_accuracy = (Sensitivity + Specificity) / 2
    Recall = Sensitivity
    if Precision + Recall ==0:
        F1_score =0
    elif Precision + Recall !=0:
        F1_score = (2 * Precision * Recall) / (Precision + Recall)

    FNR = 1-Sensitivity # 漏诊率 Missed diagnosis rate
    FPR = 1-Specificity # 误诊率 Misdiagnosis rate
    TPR = Sensitivity
    TNR = Specificity

    return Accuracy, Sensitivity, Specificity, AUC, \
           Precision, Balanced_accuracy, Recall, F1_score,\
           FNR, FPR, TPR, TNR



def plot_ROC_curve(all_test_label, all_score_for_AUC):

    FPR, TPR, thresholds = roc_curve(all_test_label, all_score_for_AUC)
    plt.plot(FPR, TPR)
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.suptitle('ROC curve of classification analysis')
    plt.show()

