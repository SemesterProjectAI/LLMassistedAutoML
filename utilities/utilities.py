import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef
)


def metrics_display(y_test, y_pred, y_pred_proba):
    # Obtain confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Output classification metrics
    tn, fp, fn, tp = cm.ravel()

    print(f'ROC_AUC score: {roc_auc_score(y_test, y_pred_proba):.3f}')
    print(f'f1 score: {f1_score(y_test, y_pred):.3f}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
    print(f'Precision: {precision_score(y_test, y_pred) * 100:.2f}%')
    print(f'Detection rate: {recall_score(y_test, y_pred) * 100:.2f}%')
    print(f'False alarm rate: {fp / (tn + fp) * 100}%')
    print(f'MCC: {matthews_corrcoef(y_test, y_pred):.2f}')

    # Display confusion matrix
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, values_format='.5g', colorbar=False)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()


def suggest_initial_search_space():
    prompt = f"""
    Given your understanding of XGBoost and general best practices in machine learning, suggest an 
    initial search space for hyperparameters. 

    Tunable hyperparameters include:
    - n_estimators (integer): Number of boosting rounds or trees to be trained.
    - max_depth (integer): Maximum tree depth for base learners.
    - min_child_weight (integer or float): Minimum sum of instance weight (hessian) needed in a leaf node. 
    - gamma (float): Minimum loss reduction required to make a further partition on a leaf node of the tree.
    - scale_pos_weight (float): Balancing of positive and negative weights.
    - learning_rate (float): Step size shrinkage used during each boosting round to prevent overfitting. 
    - subsample (float): Fraction of the training data sampled to train each tree. 
    - colsample_bylevel (float): Fraction of features that can be randomly sampled for building each level (or depth) of the tree.
    - colsample_bytree (float): Fraction of features that can be randomly sampled for building each tree. 
    - reg_alpha (float): L1 regularization term on weights. 
    - reg_lambda (float): L2 regularization term on weights. 

    The search space is defined as a dict with keys being hyperparameter names, and values 
    are the search space associated with the hyperparameter. For example:
        search_space = {{
            "learning_rate": loguniform(1e-4, 1e-3)
        }}

    Available types of domains include: 
    - scipy.stats.uniform(loc, scale), it samples values uniformly between loc and loc + scale.
    - scipy.stats.loguniform(a, b), it samples values between a and b in a logarithmic scale.
    - scipy.stats.randint(low, high), it samples integers uniformly between low (inclusive) and high (exclusive).
    - a list of possible discrete value, e.g., ["a", "b", "c"]

    Please first briefly explain your reasoning, then provide the configurations of the initial 
    search space. Enclose your suggested configurations between markers 
    [BEGIN] and [END], and assign your configuration to a variable named search_space.
    """

    return prompt


def suggest_refine_search_space(top_n, last_run_best_score, all_time_best_score):
    prompt = f"""
    Given your previously suggested search space, the obtained top configurations with their 
    test scores:
    {top_n}

    The best score from the last run was {last_run_best_score}, while the best score ever 
    achieved in all previous runs is {all_time_best_score}

    Remember, tunable hyperparameters are: n_estimators, max_depth, min_child_samples, gamma, 
    scale_pos_weight, learning_rate, subsample, colsample_bylevel, colsample_bytree, reg_alpha, 
    and reg_lambda.

    Given the insights from the search history, your expertise in ML, and the need to further 
    explore the search space, please suggest refinements for the search space in the next optimization round. 
    Consider both narrowing and expanding the search space for hyperparameters where appropriate.

    For each recommendation, please:
    1. Explicitly tie back to any general best practices or patterns you are aware of regarding XGBoost tuning
    2. Then, relate to the insights from the search history and explain how they align or deviate from these 
    practices or patterns.
    3. If suggesting an expansion of the search space, please provide a rationale for why a broader range could 
    be beneficial.


    Briefly summarize your reasoning for the refinements and then present the adjusted configurations. 
    Enclose your refined configurations between markers [BEGIN] and [END], and assign your 
    configuration to a variable named search_space.
    """

    return prompt


def suggest_metrics(report):
    prompt = f"""
    The classification problem under investigation is based on a network intrusion detection dataset. 
    This dataset contains DOS, Probe, R2L, and U2R attack types, which are all grouped under the 
    "attack" class (label: 1). Conversely, the "normal" class is represented by label 0. 
    Below are the dataset's characteristics:
    {report}.

    For this specific inquiry, you are tasked with recommending a suitable hyperparameter optimization 
    metric for training a XGBoost model. It is crucial that the model should accurately identify genuine 
    threats (attacks) without raising excessive false alarms on benign activities. They are equally important.
    Given the problem context and dataset characteristics, suggest only the name of one of the built-in 
    metrics: 
    - 'accuracy'
    - 'roc_auc' (ROCAUC score)
    - 'f1' (F1 score)
    - 'balanced_accuracy' (It is the macro-average of recall scores per class or, equivalently, raw 
    accuracy where each sample is weighted according to the inverse prevalence of its true class) 
    - 'average_precision'
    - 'precision'
    - 'recall'
    - 'neg_brier_score'


    Please first briefly explain your reasoning and then provide the recommended metric name. 
    Your recommendation should be enclosed between markers [BEGIN] and [END], with standalone string for 
    indicating the metric name.
    Do not provide other settings or configurations.
    """

    return prompt


def data_report(df, num_feats, bin_feats, nom_feats):
    # Last column is the label
    target = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    # General dataset info
    num_instances = len(df)
    num_features = features.shape[1]

    # Label class analysis
    class_counts = target.value_counts()
    class_distribution = class_counts / num_instances
    if any(class_distribution < 0.3) or any(class_distribution > 0.7):
        class_imbalance = True
    else:
        class_imbalance = False

    # Create a text report
    report = f"""Data Characteristics Report:

- General information:
  - Number of Instances: {num_instances}
  - Number of Features: {num_features}

- Class distribution analysis:
  - Class Distribution: {class_distribution.to_string()}
  {'Warning: Class imbalance detected.' if class_imbalance else ''}

- Feature analysis:
  - Feature names: {features.columns.to_list()}
  - Number of numerical features: {len(num_feats)}
  - Number of binary features: {len(bin_feats)}
  - Binary feature names: {bin_feats}
  - Number of nominal features: {len(nom_feats)}
  - Nominal feature names: {nom_feats}
"""

    return report


def extract_logs(results):
    df = pd.DataFrame(results.cv_results_)

    # Number of top-performing configurations you want to extract
    top_n = 5

    # 1. Identify top-performing configurations using rank_test_score
    top_configs = df.nsmallest(top_n, 'rank_test_score').reset_index(drop=True)

    hyperparameter_columns = [
        'param_colsample_bylevel', 'param_colsample_bytree', 'param_gamma',
        'param_learning_rate', 'param_max_depth', 'param_min_child_weight',
        'param_n_estimators', 'param_reg_alpha', 'param_reg_lambda',
        'param_scale_pos_weight', 'param_subsample'
    ]

    # Extracting the top-N configurations as strings
    config_strings = []
    for index, row in top_configs.iterrows():
        config_str = ', '.join([f"{col[6:]}: {row[col]}" for col in hyperparameter_columns])
        config_strings.append(f"Configuration {index + 1} ({row['mean_test_score']:.4f} test score): {config_str}")

    # Joining them together for a complete summary
    top_config_summary = '\n'.join(config_strings)

    # Best test score
    last_run_best_score = top_configs.loc[0, 'mean_test_score']

    return top_config_summary, last_run_best_score
