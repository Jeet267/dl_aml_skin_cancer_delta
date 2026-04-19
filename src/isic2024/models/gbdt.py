# LightGBM / XGBoost / CatBoost wrappers (sklearn API)
def get_lightgbm():
    """Uses min_sum_hessian_in_leaf=10 to fix imbalance handling."""
    pass

def get_xgboost():
    """Best single model; max_delta_step=1 critical."""
    pass

def get_catboost():
    """Uses auto_class_weights=SqrtBalanced."""
    pass
