from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Literal, Union
import matplotlib.pyplot as plt
import plotnine as pn
from lightgbm import log_evaluation
from lightgbm.callback import early_stopping
import shap, seaborn as sns



# Model Training and Hyperaparameter tuning
def tune_parameters(model, X, y, param_grid, scorer='f1', cv=5):
    gcv = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scorer)
    gcv.fit(X, y)
    return gcv


def get_feature_importance_scores(model, columns=None):
    if columns is None:
        if hasattr(model, 'feature_names_in_'):
            columns = model.feature_names_in_ 
        elif hasattr(model, 'feature_names_'):
            columns = model.feature_names_

    if hasattr(model, 'coef_'):
        varimp = model.coef_.squeeze()
    elif hasattr(model, 'named_steps'):
        model_name = list(model.named_steps.keys())[-1]
        varimp = model.named_steps[model_name].coef_.squeeze()
    elif hasattr(model, 'feature_importances_'):
        varimp = model.feature_importances_/model.feature_importances_.sum()
    return pd.Series(varimp, columns).sort_values(ascending=False)

def topn_importance(model, topn=20, columns=None):
    varimp = get_feature_importance_scores(model, columns).sort_values(ascending=True)

    if topn is not None:
        varimp = varimp[varimp.abs().nlargest(topn).index]
    varimp = varimp[varimp != 0] 
    varimp = varimp.reset_index(name='scores')
    varimp = varimp.rename(columns={'index':'name'})

    fig = (
        pn.ggplot(varimp, pn.aes('reorder(name, scores)', 'scores')) +
        pn.geom_col(fill='indianred') +
        pn.geom_text(pn.aes(label='scores'), fontweight='bold', size=8, 
                     format_string='{:.3f}', color='#2C2D2D', 
                     nudge_y=max(varimp.iloc[:, 1])*0.05) + 
        pn.theme_bw() +
        pn.theme(panel_grid=pn.element_blank(),
                 figure_size=(9,6),
                 axis_title=pn.element_text(size=9),
                 axis_text=pn.element_text(size=8),
                 axis_text_x = pn.element_blank(),
                 axis_ticks_x = pn.element_blank(),
                 plot_title=pn.element_text(face='bold', hjust=0, size=12)) +
        pn.coord_flip() +
        pn.labs(title=model.__class__.__name__, 
                x='', y='scores')
        )
    return(fig)

class OneHotEncoder:
    def __init__(self, df=None, categorical_cols:list=None, drop_first=False):
        self.df = df
        self.is_fitted = False
        self.drop_first = drop_first
        self.categorical_cols = categorical_cols
        self.categorical_features_out = []
        self.categorical_cols_unique = {}
        
    def fit(self, df=None):
        if df is None and self.df is None:
            raise ValueError('df must not be None')
        if df is not None:
            self.df = df
        if self.categorical_cols:
            if not isinstance(self.categorical_cols, (list,tuple)):
                raise TypeError(f'Categorical variables must be in a list or tuple')
        
        if self.categorical_cols is None and self.df is not None:
            self.categorical_cols = self.df.select_dtypes('O').columns.tolist()    
        
        for categorical_col in self.categorical_cols:
            self.categorical_cols_unique[categorical_col] = sorted(self.df[categorical_col].unique().tolist())
        self.is_fitted = True
        return self
    
    def transform(self, df):
        if df is None:
            raise ValueError('df must not be None')
        if self.is_fitted:
            df_new = df.copy()
            # loop through categorical columns, get unique values
            # loop through unique values, create new column, and drop categorical column
            for categorical_col in self.categorical_cols:
                unique_vals = self.categorical_cols_unique.get(categorical_col)
                if self.drop_first: # dummy encoding (for linear models)
                    unique_vals = unique_vals[1:]
                for val in unique_vals:
                    self.categorical_features_out.append(f'{categorical_col}_{val}')
                    df_new[f'{categorical_col}_{val}'] = np.where(df[categorical_col] == val, 1, 0)
        else:
            raise Exception('Encoder not fitted!')
        df_new = df_new.drop(columns=self.categorical_cols)
        return df_new
    
    def fit_transform(self, df):
        if self.df is not None:
            return self.fit(self.df).transform(df)
        else:
            return self.fit(df).transform(df)

class Model:
    def __init__(self):
        self.is_fitted = False
        self.model = None
    
    def train(self, model, X, y, eval_set=None, epochs=50, verbose_iter=200, eval_metric='rmse'):
        self.model = model
        self.is_fitted = True
        self.eval_set = eval_set
        if self.eval_set is None:
            self.model.fit(X, y)
        elif self.model.__class__.__name__ in ['LGBMClassifier', 'LGBMRegressor'] and self.eval_set is not None:
            self.model.fit(
                X, y, 
                eval_set=self.eval_set, 
                eval_metric=eval_metric,
                callbacks=[
                    early_stopping(epochs, first_metric_only=True, verbose=True),
                    log_evaluation(period=verbose_iter, show_stdv=False)
                ]
        )
        elif self.model.__class__.__name__ in ['CatBoostClassifier', 'XGBClassifier', 'CatBoostRegressor', 'XGBRegressor'] and self.eval_set is not None:
            self.model.fit(X, y, eval_set=self.eval_set, verbose=verbose_iter)
        else:
            self.model.fit(X, y)
        return self.model
        
    def predict(self, X):
        if self.is_fitted:
            return self.model.predict(X)
        else:
            raise Exception(f'{self.model.__class__.__name__} not fitted')

    def predict_proba(self, X):
        if self.is_fitted:
            return self.model.predict_proba(X)
        else:
            raise Exception(f'{self.model.__class__.__name__} not fitted')


def cross_validate_scores(model, X, y, cv, groups=None, scoring:Literal['f1', 'auc', 'rmse', 'mae', callable]='f1'):
    scores = []
    splits = list(cv.split(X,y, groups=groups))
    for xtr_idx, xva_idx in tqdm(splits):
        xtrain, ytrain = X.iloc[xtr_idx], y.iloc[xtr_idx]
        xval, yval = X.iloc[xva_idx], y.iloc[xva_idx]
        clf = Model()
        clf.train(model, xtrain, ytrain, eval_set=[(xval, yval)])
        preds = clf.predict(xval)
        if hasattr(model, 'predict_proba'):
            probs = clf.predict_proba(xval)[:,1]

        if scoring == 'f1':
            f1 = metrics.f1_score(yval, preds)
            scores.append(f1)
        elif scoring == 'auc':
            auc = metrics.roc_auc_score(yval, probs)
            scores.append(auc)
        elif scoring == 'rmse':
            rmse = metrics.root_mean_squared_error(yval, preds)
            scores.append(rmse)
        elif scoring == 'mae':
            mae = metrics.mean_absolute_error(yval, preds)
            scores.append(mae)
        elif callable(scoring):
            scorer = metrics.make_scorer(scoring)
            res = scorer(clf, yval, preds)
            scores.append(res)

        else:
            print(f'{scoring} not recognised. Must be either f1|auc|rmse|mae')
    return np.array(scores)

# SHAP 
def get_shap_values(model, df):
    explainer = shap.Explainer(model)
    shap_vals = explainer(df)
    return shap_vals[..., 1] if len(shap_vals.shape) == 3 else shap_vals


def get_permuatation_scores(model, df, seed=None, npermutations=10):
    explainer = shap.explainers.Permutation(model.predict, df, seed=seed)
    shap_vals = explainer.shap_values(df, npermutations=npermutations)
    return np.abs(shap_vals)


def plot_shap_feature_importance(shap_values:np.ndarray, feature_names, topn=25, title=None):
    shap_val_mean = np.abs(shap_values.values).mean(0) if hasattr(shap_values, 'values') else np.abs(shap_values).mean(0)
    assert len(feature_names) == len(shap_val_mean)
    varimp = pd.Series(shap_val_mean, feature_names)

    if topn is not None:
        varimp = varimp.nlargest(topn)
    varimp = varimp[varimp != 0] 
    varimp = varimp.reset_index(name='scores')
    varimp = varimp.rename(columns={'index':'name'})

    fig = (
        pn.ggplot(varimp, pn.aes('reorder(name, scores)', 'scores')) +
        pn.geom_col(fill='indianred') +
        pn.geom_text(pn.aes(label='scores'), fontweight='bold', size=8, 
                     format_string='{:.3e}', color='#2C2D2D', 
                     nudge_y=max(varimp.iloc[:, 1])*0.05) + 
        pn.theme_bw() +
        pn.theme(panel_grid=pn.element_blank(),
                 figure_size=(9,6),
                 axis_title=pn.element_text(size=9),
                 axis_text=pn.element_text(size=8),
                 axis_text_x = pn.element_blank(),
                 axis_ticks_x = pn.element_blank(),
                 plot_title=pn.element_text(face='bold', hjust=0, size=12)) +
        pn.coord_flip() +
        pn.labs(title=title, x='', y='mean(|shap values|)')
        )
    return(fig)


def plot_shap_summary(shap_values, X, plot_type='dot', max_display=10, show=True):
    shap.summary_plot(shap_values, X, plot_type=plot_type, 
                      max_display=max_display, show=show)


## Evaluation metrics

def eval_metrics(model, X, y, task_type:Literal['classification', 'regression']='regression', transform_type=None):
    if task_type == 'regression':
        scores = regression_eval_metrics(model, X, y, transform_type=transform_type)
    elif task_type == 'classification':
        scores = classification_eval_metrics(model, X, y)
    
    return scores

def classification_eval_metrics(model, X, y, average_type:Literal['weighted','micro', 'macro']='macro'):
    preds = model.predict(X)
    unique_class = np.unique(y)
    if len(unique_class) == 2:
        probs = model.predict_proba(X)[:,1]
        auc = metrics.roc_auc_score(y, probs, )
        rec = metrics.recall_score(y, preds)
        prec = metrics.precision_score(y, preds)
        f1 = metrics.f1_score(y, preds)
        spec = specificity(preds, y, unique_class, average_type='binary')
        
    elif len(unique_class) > 2:
        probs = model.predict_proba(X)
        auc = metrics.roc_auc_score(y, probs, average=average_type, multi_class='raise')
        rec = metrics.recall_score(y, preds, average=average_type)
        prec = metrics.precision_score(y, preds, average=average_type)
        f1 = metrics.f1_score(y, preds, average=average_type)
        spec = specificity(preds, y, unique_class, average_type=average_type)

    acc = np.mean(y == preds)
    

    scores = pd.DataFrame([acc, auc, rec, prec, f1, spec], 
                          columns=['scores'], 
                          index=['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Specificity'])
    return scores.T



def regression_eval_metrics(model, X, y, transform_type=None):
    preds = model.predict(X)
    if transform_type == 'log1p':
        y = np.expm1(y)
        preds = np.expm1(preds)
    elif transform_type == 'sqrt':
        y = np.square(y)
        preds = np.square(preds)

    
    mae = metrics.mean_absolute_error(y, preds)
    mse = metrics.mean_squared_error(y, preds)
    rsq = metrics.r2_score(y, preds)
    rmse = metrics.root_mean_squared_error(y, preds)
    scores = pd.DataFrame([rmse, mae, mse, rsq], 
                          columns=['scores'], 
                          index=['RMSE', 'MAE', 'MSE', 'Rsq'])
    return scores.T


def specificity(preds, 
                actual, 
                unique_class=None, 
                positive_val = 1,
                average_type:Literal['weighted','micro', 'macro', 'binary']='weighted'):
        # class specificity tn / (tn+fp)
        if unique_class is None:
             unique_class = sorted(np.unique(actual))

        if len(unique_class) == 2:
            average_type = 'binary'
            return np.sum((preds != positive_val) & (actual != positive_val)) / np.sum(actual != positive_val)
        elif len(unique_class) > 2:
            tn = np.array([np.sum((preds != i) & (actual != i)) for i in sorted(unique_class)])
            fp = np.array([np.sum((preds != i) & (actual == i)) for i in sorted(unique_class)])
            score = tn / (tn+fp)
            class_counts = np.array([np.sum(actual==i) for i in sorted(unique_class)])
        
            if average_type == 'weighted':
                return np.average(score, weights=class_counts)
            elif average_type == 'macro':
                return np.mean(score)
            elif average_type == 'micro':
                return np.sum(tn) / np.sum(tn+fp)
            else:
                print(f'{average_type} not valid')
                exit(1)


# Model evaluation visualisation
def print_classification_report(model, X, y, display_names=None):
    preds = model.predict(X)
    print(metrics.classification_report(y, preds, target_names=display_names))


def regression_performance_chart_report(model, X, y):
    predictions = model.predict(X)
    df = pd.DataFrame({'Actual':y, 'Predictions':predictions, 'Errors': y-predictions})
    cor_result = np.corrcoef(df['Actual'], df['Predictions'])[0][1]
    fig, ax = plt.subplots(1,2,figsize=(9,5))

    sns.scatterplot(df, x='Actual', y='Predictions', color='indianred', alpha=0.8, ax=ax[0])
    sns.scatterplot(df, x='Predictions', y='Errors', color='steelblue', alpha=0.8, ax=ax[1])
    ax[1].axhline(0, color='#2C2D2D', linestyle='--', lw=0.8)
    ax[0].set(xlabel='Actual values', ylabel='Predicted values')
    ax[1].set(xlabel='Predicted values', ylabel='Errors')
    ax[0].set_title('Actual vs Predicted values', fontsize=10)
    ax[1].set_title('Residual Plot', fontsize=10)
    ax[0].legend([f'r = {cor_result:.3f}'], loc='lower right')
    plt.suptitle(model.__class__.__name__, x=0.12, y=.96, fontweight='bold')
    plt.tight_layout()


def classification_performance_chart_report(model, X, y, display_names=None):
    display_labs = y.drop_duplicates()

    fig, ax = plt.subplots(1,3, figsize=(12,5))
    _ = metrics.RocCurveDisplay.from_estimator(model, X, y, ax=ax[0])
    _ = metrics.PrecisionRecallDisplay.from_estimator(model, X, y, ax=ax[1])
    _ = metrics.ConfusionMatrixDisplay.from_estimator(model, X, y, display_labels=display_labs if display_names is None else display_names, 
                                                      ax=ax[2], cmap ='Greens', colorbar=False)
    ax[0].set_title('ROC curve', fontsize=10)
    ax[1].set_title('Precision-Recall curve', fontsize=10)
    ax[2].set_title('Confusion Matrix', fontsize=10)
    plt.suptitle(model.__class__.__name__, x=0.15, y=0.90, fontweight='bold')
    plt.tight_layout()




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


def run_conditional_permutation_type(model, X, y, corr_vals, feature, feature_corr_vars, 
                                     task_type, scoring, seed, n_permutations):
    baseline_scores = eval_metrics(model, X, y, task_type)
    baseline_scores.columns = baseline_scores.columns.str.lower()
    if scoring in baseline_scores.columns:
        baseline_scores = baseline_scores[scoring].values
    else: 
        raise TypeError (f"{scoring} not recognised. Scoring must be in {baseline_scores.columns.str.lower().tolist()}")
        exit()
    X_shuffled = X.copy()

    rng = np.random.RandomState(seed)
    X_shuffled[feature_corr_vars] = corr_vals

    score_difference = []
    for _ in range(n_permutations):
        X_shuffled[feature] = rng.permutation(X_shuffled[feature])
        permuted_scores = eval_metrics(model, X_shuffled, y, task_type)
        permuted_scores.columns = permuted_scores.columns.str.lower()
        permuted_scores = permuted_scores[scoring].values
        diff = baseline_scores - permuted_scores
        score_difference.append(diff)
    return np.mean(score_difference), np.std(score_difference)


def run_joint_permutation_type(model, X, y, feature, feature_corr_vars, task_type, scoring, seed, n_permutations):
    baseline_scores = eval_metrics(model, X, y, task_type)
    baseline_scores.columns = baseline_scores.columns.str.lower()
    if scoring in baseline_scores.columns:
        baseline_scores = baseline_scores[scoring].values
    else: 
        raise TypeError (f"{scoring} not recognised. Scoring must be in {baseline_scores.columns.tolist()}")
        exit()
    
    X_shuffled = X.copy()
    rng = np.random.RandomState(seed)
    score_difference = []
    for _ in range(n_permutations):
        if len(feature_corr_vars) > 0:
            X_shuffled[[feature]+feature_corr_vars] = rng.permutation(X_shuffled[[feature]+feature_corr_vars])
        else:
            X_shuffled[feature] = rng.permutation(X_shuffled[feature])
        permuted_scores = eval_metrics(model, X_shuffled, y, task_type)
        permuted_scores.columns = permuted_scores.columns.str.lower()
        permuted_scores = permuted_scores[scoring]
        diff = baseline_scores - permuted_scores
        score_difference.append(diff)
    return np.mean(score_difference), np.std(score_difference)



def permutation_importance(model, X, y, Xval=None, yval=None, cv:Union[None, int]=None, 
                           groups=None, threshold=0.85, n_permutations=20, seed=42, 
                           task_type:Literal['regression', 'classification']='classification',
                           check_correlated_variables:bool=True,
                           numerical_features = None, scoring=None, test_size=0.2, 
                           value_type:Literal['mean', 'median']='mean',
                           permutation_type:Literal['conditional', 'joint', None]=None
                           ):
    """
    Computes baseline model performance and computes the drop in performance after randomly shuffling a feature. Permutation type could be "conditional", "joint", or None.

    In None, no multicollinearity between variables is considered while in conditional and joint, multicollinearity is considered. 
    In conditional, a feature is permuted while the values of its correlated features are replaced with either their mean or median values
    In joint, a feature, alongside its correlated features are jointly shuffled and the drop in performance is calculated.
    In joint, fewer variables are returned where permutation on multicollinear features is done once.
    
    Parameters
    ==========
    model: Classification or Regression model
    X, y: Training data
    cv: If cross validation should be done. Int value or None. If None, no cross-validation is performed
    groups: If cross-validation, groups to split data (Check GroupKFold or StratifiedGroupKFold)
    threshold: Threshold to select multicollinear variables
    n_permutations: Number of times to permute a feature
    seed: Pseudo-random number for reproducibility
    task_type: If task is classification or regression type
    check_correlated_variables: Boolean. To indicate if correlated variables should be checked
    scoring: Metric to use for evaluation
    permutation_type: Type of permutation importance to perform (conditional, joint or None)
    """
    from utils.edatools import select_correlated_features
    
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
    if not isinstance(Xtrain_, list):
        model.fit(Xtrain_, ytrain_)
        for feature in tqdm(features, desc='Permutation Importance'):
            if check_correlated_variables:
                feature_corr_vars = corr_df.query(f'Feature1 == "{feature}"').Feature2.tolist() # get correlated variables
                if len(feature_corr_vars) > 0: # not empty
                    if permutation_type == 'conditional': 
                        # get mean or median values of correlated features
                        corr_vals = Xtrain_[feature_corr_vars].mean() if value_type == 'mean' else Xtrain_[feature_corr_vars].median()
                        mean_score, std_score = run_conditional_permutation_type(model, Xtest_, ytest_, corr_vals, feature, feature_corr_vars, 
                                                                                task_type, scoring, seed, n_permutations)
                    elif permutation_type == 'joint':
                        mean_score, std_score = run_joint_permutation_type(model, Xtest_, ytest_, feature, feature_corr_vars, 
                                                                           task_type, scoring, seed, n_permutations)
                    else:
                        print(f'{permutation_type} not recognised!')
                        exit(1)
                else:
                    # if no correlation features exist
                    feature_corr_vars = [] # set to empty so that correlated features are not jointly permuted with feature
                    mean_score, std_score = run_joint_permutation_type(model, Xtest_, ytest_, feature, feature_corr_vars, 
                                                                       task_type, scoring, seed, n_permutations)
            elif permutation_type is None: 
                feature_corr_vars = [] # set to empty so that correlated features are not jointly permuted with feature
                mean_score, std_score = run_joint_permutation_type(model, Xtest_, ytest_, feature, feature_corr_vars, 
                                                                   task_type, scoring, seed, n_permutations)
            results.append([mean_score, std_score])
    elif isinstance(Xtrain_, list): # cross validation
        cv_result = []
        for i in tqdm(range(len(Xtrain_)), desc='Permutation Importance (crossvalidation)'):
            xtr, xte, ytr, yte = Xtrain_[i], Xtest_[i], ytrain_[i], ytest_[i]
            model.fit(xtr, ytr)
            feature_result = []
            for feature in features:
                if check_correlated_variables:
                    feature_corr_vars = corr_df.query(f'Feature1 == "{feature}"').Feature2.tolist() # get correlated variables
                    if len(feature_corr_vars) > 0: # not empty
                        if permutation_type == 'conditional': 
                            # get mean or median values of correlated features
                            corr_vals = xtr[feature_corr_vars].mean() if value_type == 'mean' else xtr[feature_corr_vars].median()
                            mean_score, std_score = run_conditional_permutation_type(model, xte, yte, corr_vals, feature, feature_corr_vars, 
                                                                                    task_type, scoring, seed, n_permutations)
                            # feature_result.append(mean_score)
                        elif permutation_type == 'joint':
                            mean_score, std_score = run_joint_permutation_type(model, xte, yte, feature, feature_corr_vars, 
                                                                            task_type, scoring, seed, n_permutations)
                            # feature_result.append(mean_score)
                        else:
                            print(f'{permutation_type} not recognised!')
                            exit(1)
                    else:
                        # if no correlation features exist

                        mean_score, std_score = run_joint_permutation_type(model, xte, yte, feature, feature_corr_vars, 
                                                                           task_type, scoring, seed, n_permutations)
                elif permutation_type is None: 
                    feature_corr_vars = [] # set to empty so that correlated features are not jointly permuted with feature
                    mean_score, std_score = run_joint_permutation_type(model, xte, yte, feature, feature_corr_vars, 
                                                                    task_type, scoring, seed, n_permutations)
                feature_result.append(mean_score)
            cv_result.append(feature_result)
        cv_result = np.array(cv_result)
        results = np.c_[np.mean(cv_result, axis=0), np.std(cv_result, axis=0)]
    results = pd.DataFrame(results, columns=['mean', 'std'], index=features)
    return results.sort_values(by='mean', ascending=False)