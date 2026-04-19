# pAUC (exact competition impl), ECE, Brier
def exact_isic_pauc(y_true, y_pred, min_tpr=0.80):
    """
    Metric: partial AUC above 80% TPR, range [0.0, 0.2].
    Uses label-flip + prediction negation.
    NOT sklearn roc_auc_score which applies McClish normalisation.
    """
    pass

def expected_calibration_error():
    pass

def brier_score():
    pass
