from sklearn import metrics
import pandas as pd
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
import seaborn as sns


## Evaluation metrics
def eval_metrics(model, X, y, task_type:Literal['classification', 'regression']='regression', 
                 transform_type=None, average_type:Literal['weighted','micro', 'macro']='macro'):
    if task_type == 'regression':
        scores = regression_eval_metrics(model, X, y, transform_type)
    elif task_type == 'classification':
        scores = classification_eval_metrics(model, X, y, average_type)
    
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
        auc = metrics.roc_auc_score(y, probs, average=average_type, multi_class='ovr')
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


def specificity(preds, actual, unique_class=None, positive_val = 1,
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
