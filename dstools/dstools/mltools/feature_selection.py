from dstools.mltools import evaluation
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Literal, Union, List


# Permutation Importance
def split_data_permutation(X, y, Xval=None, yval= None, cv=None, groups=None, 
                           seed=None, test_size=0.2,
                           task_type:Literal['regression', 'classification']='classification'):
    """
    Splits data into holdout test that will be used to calculate permutation importance
    """
    from sklearn.model_selection import StratifiedKFold, KFold

    xtrain_, xtest_, ytrain_, ytest_ = None, None, None, None
    if Xval is not None and yval is not None:
        xtrain_, xtest_, ytrain_, ytest_ = X, Xval, y, yval
    elif cv is None:
        from sklearn.model_selection import train_test_split
        if task_type == 'regression':
            xtrain_, xtest_, ytrain_, ytest_ = train_test_split(X, y, random_state=seed, test_size=test_size)
        elif task_type == 'classification':
            xtrain_, xtest_, ytrain_, ytest_ = train_test_split(X, y, random_state=seed, stratify=y, test_size=test_size)
    elif type(cv) is int:
        if task_type == 'regression':
            cv_fold = KFold(n_splits=cv, random_state=seed, shuffle=True)
        elif task_type == 'classification':
            cv_fold = StratifiedKFold(n_splits=cv, random_state=seed, shuffle=True)
    elif hasattr(cv, 'split'): # sklearn's split class
        cv_fold = cv
        
        xtrain_, xtest_, ytrain_, ytest_ = [], [], [], []
        for tr_idx, val_idx in cv_fold.split(X, y, groups=groups):
            xtrain_.append(X.iloc[tr_idx])
            ytrain_.append(y.iloc[tr_idx])
            xtest_.append(X.iloc[val_idx])
            ytest_.append(y.iloc[val_idx])
    return xtrain_, xtest_, ytrain_, ytest_


def run_conditional_permutation_type(model, X, y, corr_vals, feature, 
                                     feature_corr_vars, task_type, scoring, 
                                     baseline_scores, seed, n_permutations) -> List[float, float]:
    X_shuffled = X.copy()

    rng = np.random.RandomState(seed)
    X_shuffled[feature_corr_vars] = corr_vals

    score_difference = []
    for _ in range(n_permutations):
        X_shuffled[feature] = rng.permutation(X_shuffled[feature])
        permuted_scores = evaluation.eval_metrics(model, X_shuffled, y, task_type)
        permuted_scores.columns = permuted_scores.columns.str.lower()
        permuted_scores = permuted_scores[scoring].values
        diff = baseline_scores - permuted_scores
        score_difference.append(diff)
    return np.mean(score_difference), np.std(score_difference)


def get_baseline_score(model, X, y, task_type, scoring) -> float:
    """
    Returns baseline scores of model evaluated on holdout test set
    """
    scores = evaluation.eval_metrics(model, X, y, task_type)
    scores.columns = scores.columns.str.lower()
    if scoring in scores.columns:
        scores = scores[scoring].values
        return scores
    else: 
        raise TypeError (f"{scoring} not recognised. Scoring must be in {scores.columns.tolist()}")
        exit()
    

def run_joint_permutation_type(model, X, y, feature, feature_corr_vars, task_type, 
                               scoring, baseline_scores, seed, n_permutations) -> List[float, float]:
    X_shuffled = X.copy()
    rng = np.random.RandomState(seed)
    score_difference = []
    for _ in range(n_permutations):
        if len(feature_corr_vars) > 0:
            X_shuffled[[feature]+feature_corr_vars] = rng.permutation(X_shuffled[[feature]+feature_corr_vars])
        else:
            X_shuffled[feature] = rng.permutation(X_shuffled[feature])
        permuted_scores = evaluation.eval_metrics(model, X_shuffled, y, task_type)
        permuted_scores.columns = permuted_scores.columns.str.lower()
        permuted_scores = permuted_scores[scoring]
        diff = baseline_scores - permuted_scores
        score_difference.append(diff)
    return np.mean(score_difference), np.std(score_difference)


def get_permuted_feature_scores(model, Xtrain, X, y, corr_df, feature, permutation_type, 
                                task_type, scoring, baseline_scores, seed, n_permutations, 
                                value_type, check_correlated_variables) -> List[float, float]:
    """
    Calculates scores of permuted feature

    :param model: Fitted model
    :param Xtrain: Training data for calculating mean or median score for other correlated features during conditional permutation importance
    :param X, y: Holdout data for evaluating importance of shuffled feature
    :param corr_df: DataFrame of all correlated features
    :param feature: Feature name to be permuted
    :param permutation type: Conditional|Joint|None
    :param task_type: Classification|Regression
    :param scoring: Evaluation metric
    :param baseline_scores: Baseline score of fitted model
    :param seed: PseudoRandom number for reproducibility
    :param n_permutations: Number of times to permute feature
    :param value_type: Mean|Median. Value to replace correlated features
    :param check_correlated_variables: Boolean. To check for correlated variables

    :Returns: performance of permuted feature
    """
    if check_correlated_variables:
        feature_corr_vars = corr_df.query(f'Feature1 == "{feature}"').Feature2.tolist() # get correlated variables
        if len(feature_corr_vars) > 0: # not empty
            if permutation_type == 'conditional': 
                # get mean or median values of correlated features
                corr_vals = Xtrain[feature_corr_vars].mean() if value_type == 'mean' else Xtrain[feature_corr_vars].median()
                mean_score, std_score = run_conditional_permutation_type(model, X, y, corr_vals, feature, feature_corr_vars, 
                                                                         task_type, scoring, baseline_scores, seed, n_permutations)
            elif permutation_type == 'joint':
                mean_score, std_score = run_joint_permutation_type(model, X, y, feature, feature_corr_vars, 
                                                                   task_type, scoring, baseline_scores, seed, 
                                                                   n_permutations)
            else:
                print(f'{permutation_type} not recognised!')
                exit(1)
        else:
            # if no correlation features exist
            mean_score, std_score = run_joint_permutation_type(model, X, y, feature, feature_corr_vars, task_type, 
                                                               scoring, baseline_scores, seed, n_permutations)
    elif permutation_type is None: 
        feature_corr_vars = [] # set to empty so that correlated features are not jointly permuted with feature
        mean_score, std_score = run_joint_permutation_type(model, X, y, feature, feature_corr_vars, task_type, 
                                                           scoring, baseline_scores, seed, n_permutations)
    return mean_score, std_score


def permutation_importance(model, X, y, Xval=None, yval=None, cv:Union[None, int]=None, 
                           groups=None, threshold=0.85, n_permutations=20, seed=42, 
                           task_type:Literal['regression', 'classification']='classification',
                           check_correlated_variables:bool=True,
                           numerical_features = None, scoring=None, test_size=0.2, 
                           value_type:Literal['mean', 'median']='mean',
                           permutation_type:Literal['conditional', 'joint', None]=None
                           ) -> pd.DataFrame:
    """
    Computes baseline model performance and computes the drop in performance after randomly shuffling a feature. 
    Permutation type could be "conditional", "joint", or None.

    In None, no multicollinearity between variables is considered while in conditional and joint, multicollinearity is considered. 
    In conditional, a feature is permuted while the values of its correlated features are replaced with either their mean or median values
    In joint, a feature, alongside its correlated features are jointly shuffled and the drop in performance is calculated.
    In joint, fewer variables are returned where permutation on multicollinear features is done once.
    
    
    :param model: Classification or Regression model
    :param: X, y: Training data
    :param: Xval, yval: Validation data
    :param cv: If cross validation should be done. Int value or None. If None, no cross-validation is performed
    :param groups: If cross-validation, groups to split data (Check GroupKFold or StratifiedGroupKFold)
    :param threshold: Threshold to select multicollinear variables
    :param n_permutations: Number of times to permute a feature
    :param seed: Pseudo-random number for reproducibility
    :param task_type: If task is classification or regression type
    :param check_correlated_variables: Boolean. To indicate if correlated variables should be checked
    :param scoring: Metric to use for evaluation
    :param permutation_type: Type of permutation importance to perform (conditional, joint or None)
    """
    from dstools.edatools import select_correlated_features
    
    features = X.columns.tolist()

    if check_correlated_variables:
        print('==> Checking for correlated variables')
        if numerical_features is None:
            numerical_features = X.select_dtypes('number').columns.tolist()
        corr_df, corr_vars = select_correlated_features(X[numerical_features], threshold, return_df=True)
        print('==> Check completed\n')

    Xtrain_, Xtest_, ytrain_, ytest_ = split_data_permutation(X, y, Xval, yval, cv, groups, seed, test_size, task_type)
    
    results = []
    print('==> Fitting model to get baseline score')

    if not isinstance(Xtrain_, list): # if not list
        model.fit(Xtrain_, ytrain_)
        # get baseline score
        baseline_scores = get_baseline_score(model, Xtest_, ytest_, task_type, scoring)
        print(f'Baseline score: {baseline_scores:.4f}\n')
        for feature in tqdm(features, desc='Permutation Importance'):
            mean_score, std_score = get_permuted_feature_scores(model, Xtrain_, Xtest_, ytest_, corr_df, feature, 
                                                                permutation_type, task_type, scoring, baseline_scores, 
                                                                seed, n_permutations, value_type, check_correlated_variables)
            results.append([mean_score, std_score])

    elif isinstance(Xtrain_, list): # cross validation
        cv_result = []
        for i in tqdm(range(len(Xtrain_)), desc='Permutation Importance (crossvalidation)'):
            xtr, xte, ytr, yte = Xtrain_[i], Xtest_[i], ytrain_[i], ytest_[i]
            model.fit(xtr, ytr)
            baseline_scores = get_baseline_score(model, xte, yte, task_type, scoring)
            print(f'Fold {i+1} Baseline score: {baseline_scores:.4f}\n')
            feature_result = []
            for feature in features:
                mean_score, std_score = get_permuted_feature_scores(model, xtr, xte, yte, corr_df, feature, 
                                                                    permutation_type, task_type, scoring, baseline_scores, 
                                                                    seed, n_permutations, value_type, check_correlated_variables)
                feature_result.append(mean_score)
            cv_result.append(feature_result)
        cv_result = np.array(cv_result)
        results = np.c_[np.mean(cv_result, axis=0), np.std(cv_result, axis=0)]
    results = pd.DataFrame(results, columns=['mean', 'std'], index=features)
    return results.sort_values(by='mean', ascending=False)