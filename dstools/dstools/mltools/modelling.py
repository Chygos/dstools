from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Literal, Union
import plotnine as pn
from lightgbm import log_evaluation
from lightgbm.callback import early_stopping
import shap


# Model Training and Hyperaparameter tuning
def tune_parameters(model, X, y, param_grid, scorer='f1', cv=5):
    """
    Tunes model parameters using Gridsearch method
    """
    gcv = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scorer)
    gcv.fit(X, y)
    return gcv


def get_feature_importance_scores(model, columns=None):
    """
    Returns feature importance scores of fitted model

    :param model: Fitted Model
    :param columns: None|Model input variables (optional).
    """
    if hasattr(model, 'coef_'):
        varimp = model.coef_.squeeze()
    elif hasattr(model, 'named_steps'):
        model_name = list(model.named_steps.keys())[-1] # get model name
        model = model.named_steps[model_name] # reassign model and check for coef_ or feature_importances_ attributes
        if hasattr(model, 'coef_'):
            varimp = model.coef_.squeeze()
        elif hasattr(model, 'feature_importances_'):
            varimp = model.feature_importances_/model.feature_importances_.sum()    
    elif hasattr(model, 'feature_importances_'):
        varimp = model.feature_importances_/model.feature_importances_.sum()
    else:
        raise TypeError (f'{model.__class__.__name__} not fitted')
    
    if columns is None:
        if hasattr(model, 'feature_names_in_'):
            columns = model.feature_names_in_ 
        elif hasattr(model, 'feature_names_'):
            columns = model.feature_names_
    return pd.Series(varimp, columns).sort_values(ascending=False)


def topn_importance(model, topn=20, columns=None):
    """
    Visualises the top n feature importance of a fitted model.

    :param model: Fitted Model
    :param topn: Number of features to visualise
    :param columns: Optional. None or input variables of fitted model
    """
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
    """
    One-Hot encodes categorical variables
    """
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
    """
    Fits a model and returns model predictions
    """
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


def cross_validate_scores(model, X, y, cv, groups=None, scoring:Union[callable, Literal['f1', 'auc', 'rmse', 'mae']]='f1'):
    """
    Returns model performance scores based on cross-validation
    """
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
    """
    Calculates shap values of fitted model

    :param model: Fitted Model
    :param df: Pandas DataFrame for calculating shap values

    Returns shap values of fitted model
    """
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