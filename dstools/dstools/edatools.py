import numpy as np
from pandas import DataFrame, concat
from seaborn import histplot, countplot
import matplotlib.pyplot as plt
from IPython.display import display
from typing import Literal
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def select_correlated_features(df, threshold=0.85, 
                               corr_type:Literal['spearman', 'pearson', 'kendall']='spearman', 
                               return_df=False):
    """
    Checks for Multicollinearity between numerical variables

    :param df: Pandas DataFrame dataset
    :param threshold: Cutoff for selecting multicollinear variables
    :param corr_type: Spearman|Kendall|Pearson. Correlation method
    :param return_df: Boolean to return correlated features as Pandas DataFrame
    """
    if not isinstance(df, DataFrame):
        df = DataFrame(df)

    # Calculate the correlation matrix
    corr_matrix = df.corr(corr_type).abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find index of feature pairs with correlation greater than threshold
    correlated_features_df = [(column, row, upper[column][row]) 
                               for column in upper.columns 
                               for row in upper.index 
                               if upper[column][row] > threshold]
    correlated_features = [col for col in upper.columns if any(upper[col] > threshold)]
    
    # Convert to DataFrame for better readability
    correlated_features_df = DataFrame(correlated_features_df, columns=['Feature1', 'Feature2', 'scores'])
    if return_df:
        return correlated_features_df, correlated_features
    else:
        return correlated_features

def plot_numerical_features_on_top_each_other(train, test=None, numerical_columns=None):
    """
    Plots numerical features from train and test sets on top of each other

    :param train: Train dataset
    :param test: Test dataset
    :param numerical_columns: Numerical columns to check for multicollinearity
    """
    if numerical_columns is None:
        numerical_columns = train.select_dtypes('number').columns.tolist()

    col_sum = len(numerical_columns)
    ncol = 4
    nrow = (col_sum // ncol) if col_sum%ncol == 0 else (col_sum // ncol) + 1
    if col_sum == ncol: nrow = ncol = 2

    if col_sum > 10:
        figsize = (12,12) 
    else: 
        figsize = (10,8)

    if test is not None:
        df = concat([train, test])[numerical_columns]
        df = df.reset_index(drop=True)
        df['type'] = np.repeat(np.array(['Train', 'Test']), [len(train), len(test)])
    elif test is None:
        df = train.copy()
        df['type'] = 'Train'
    else:
        exit()

    # plot figure
    plt.figure(figsize=figsize)

    for i in range(col_sum):
        num_col = numerical_columns[i]
        plt.subplot(nrow, ncol, i+1)
        histplot(df, x=num_col, hue='type', bins=40, multiple='stack', 
                     kde=True, alpha=0.7, stat='percent')
    plt.tight_layout()
    plt.show()


def plot_numerical_features_side_by_side(train, test=None, numerical_columns=None):
    """
    Plots numerical features from train and test sets side by side

    :param train: Train dataset
    :param test: Test dataset
    :param numerical_columns: Numerical columns to check for multicollinearity
    """
    if numerical_columns is None:
        numerical_columns = train.select_dtypes('number').columns.tolist()

    len_numerical = len(numerical_columns)
    if len_numerical >= 5:
        figsize = (12,12) 
    else: 
        figsize = (10,8)

    if test is not None:
        plt.rcParams['font.size'] = 8
        fig, ax = plt.subplots(len_numerical, 2, figsize=figsize)
        ax[0, 0].set_title('Train Data')
        ax[0, 1].set_title('Test Data')
        for i in range(len_numerical):
            num_col = numerical_columns[i]
            histplot(train, x=num_col, ax=ax[i, 0], color='indianred', stat='percent', 
                         kde=True, bins=40, multiple='stack')
            histplot(test, x=num_col, ax=ax[i, 1], color='steelblue', stat='percent', 
                         kde=True, bins=40, multiple='stack')
        fig.tight_layout()
        plt.show()

    elif test is None:
        plt.rcParams['font.size'] = 8
        plt.figure(figsize=figsize)
        ncol = 4; nrow = (len_numerical//ncol) if (len_numerical % ncol == 0) else (len_numerical//ncol)+1
        if len_numerical == ncol: nrow = ncol = 2
        for i in range(len_numerical):
            cat_col = numerical_columns[i]
            plt.subplot(nrow, ncol, i+1)
            histplot(train, x=num_col, stat='percent', kde=True, bins=40, multiple='stack')
        plt.tight_layout()
        plt.rcdefaults()
        plt.show()
    else:
        return None


def plot_target_distribution(train, target_col, target_type=Literal['categorical', 'numerical']):
    """
    Plots target distribution

    :param train: Train dataset
    :param target_col: Target column name
    :param target_type: Data type of target variable (categorical or numerical)
    """
    if target_type == 'categorical':
        countplot(train, x=target_col, hue=target_col, stat='percent', legend=False);
    elif target_type == 'numerical':
        histplot(train, x=target_col, bins=25, stat='percent');
    else:
        return None


def plot_categorical_features_side_by_side(train, test=None, categorical_columns=None):
    """
    Plots categorical features from train and test sets side by side

    :param train: Train dataset
    :param test: Test dataset
    :param categorical_columns: Categorical column names
    """
    if categorical_columns is None:
        categorical_columns = train.select_dtypes(['category', 'object']).columns.tolist()

    len_categorical = len(categorical_columns)
    if len_categorical >= 5:
        figsize = (12,12) 
    else: 
        figsize = (10,8)

    if test is not None:
        plt.rcParams['font.size'] = 8
        fig, ax = plt.subplots(len_categorical, 2, figsize=figsize)
        ax[0, 0].set_title('Train Data')
        ax[0, 1].set_title('Test Data')
        for i in range(len_categorical):
            cat_col = categorical_columns[i]
            if train[cat_col].nunique() > 8:
                countplot(train, y=cat_col, ax=ax[i, 0], 
                          color='indianred', 
                          stat='percent', width=0.8)
                countplot(test, y=cat_col, ax=ax[i, 1], 
                          color='steelblue', 
                          stat='percent', width=0.8)
            else:
                countplot(train, x=cat_col, ax=ax[i, 0], 
                          color='indianred', 
                          stat='percent', width=0.8)
                countplot(test, x=cat_col, ax=ax[i, 1], 
                          color='steelblue', 
                          stat='percent', width=0.8)
        fig.tight_layout()
        plt.show()

    elif test is None:
        plt.rcParams['font.size'] = 8
        plt.figure(figsize=figsize)
        ncol = 4; nrow = (len_categorical//ncol) if (len_categorical % ncol == 0) else (len_categorical//ncol)+1
        if len_categorical == ncol: nrow = ncol = 2
        for i in range(len_categorical):
            cat_col = categorical_columns[i]
            plt.subplot(nrow, ncol, i+1)

            if train[cat_col].nunique() > 8:
                countplot(train, y=cat_col, stat='percent', width=0.8)
            else:
                countplot(train, x=cat_col, stat='percent', width=0.8)
            plt.title(cat_col)
        plt.tight_layout()
        plt.rcdefaults()
        plt.show()
    else:
        return None


def get_categorical_data_summary(df, col):
    """
    Gets summaries of a categorical feature

    :param df: Pandas DataFrame
    :param col: Categorical column
    """
    unique = np.unique(df[col])
    missing = df[col].isna().sum()
    missing_perc = 100*df[col].isna().mean()
    dtype_ = df[col].dtype.name
    return unique, missing, missing_perc, dtype_


def summarize_categorical_variables(train, test=None, categorical_variables=None):
    """
    Returns summary of categorical variables

    :param train: Train Dataset
    :param test: Test Dataset
    :param categorical_variables: Categorical Features (Optional)
    """
    if categorical_variables is None:
        categorical_variables = train.select_dtypes(['object', 'category']).columns.tolist()
    if categorical_variables is None and test is None:
        test_cat_cols = test.select_dtypes(['object', 'category']).columns.tolist()
        categorical_variables = list(set(test_cat_cols).intersection(
            set(categorical_variables)))
    
    result = DataFrame()
    result['Feature'] = categorical_variables
    result['Feature'] = result['Feature'].str.replace('_', ' ')
    if test is None:
        for i in tqdm(range(len(categorical_variables)), desc='Categorical Variable Summary'):
            col = categorical_variables[i]
            unique, missing, missing_perc, dtype_ = get_categorical_data_summary(train, col)
            result.loc[i, 'Train nunique values'] = len(unique)
            result.loc[i, 'Train missing values'] = missing
            result.loc[i, 'Train missing values (%)'] = missing_perc
            result.loc[i, 'Train dtypes'] = dtype_
    if test is not None:    
        for i in range(len(categorical_variables)):
            col = categorical_variables[i]
            train_unique, train_missing, train_missing_perc, train_dtype = get_categorical_data_summary(train, col)
            test_unique, test_missing, test_missing_perc, test_dtype = get_categorical_data_summary(test, col)
            result.loc[i, 'Train nunique values'] = len(train_unique)
            result.loc[i, 'Test nunique values'] = len(test_unique)
            result.loc[i, 'Train missing values'] = train_missing
            result.loc[i, 'Test missing values'] = test_missing
            result.loc[i, 'Train missing values (%)'] = train_missing_perc
            result.loc[i, 'Test missing values (%)'] = test_missing_perc
            result.loc[i, 'Train dtypes'] = train_dtype
            result.loc[i, 'Test dtypes'] = test_dtype
            result.loc[i, 'Unique values in test and not train'] = len(set(test_unique).difference(set(train_unique)))
            result.loc[i, 'Unique values in train and not test'] = len(set(train_unique).difference(set(test_unique)))
    return display(result)


def get_numerical_data_summary(df, numerical_variables):
    """
    Gets summary of numerical variables

    :param df: Pandas DataFrame
    :param numerical_variables: Numerical features
    """
    result = df[numerical_variables].describe()
    nunique = df[numerical_variables].nunique()
    missing = df[numerical_variables].isna().sum()
    iqr = df[numerical_variables].quantile(0.75) - df[numerical_variables].quantile(0.25)
    p95 = df[numerical_variables].quantile(0.95)
    skew = df[numerical_variables].skew()
    kurt = df[numerical_variables].kurt()

    result.loc['IQR', :] = iqr
    result.loc['p95', :] = p95
    result.loc['skew', :] = skew
    result.loc['kurt', :] = kurt
    result.loc['nunique', :] = nunique
    result.loc['nmissing', : ] = missing
    result.columns = result.columns.str.replace('_', ' ')
    return result

def summarize_numerical_variables(train, test=None, numerical_columns=None):
    """
    Returns summary of numerical variables

    :param train: Train Dataset
    :param test: Test Dataset
    :param numerical_variables: Numerical Features (Optional)
    """
    numerical_variables = numerical_columns
    if numerical_columns is None:
        numerical_variables = train.select_dtypes('number').columns.tolist()
    if numerical_columns is None and test is not None:
        test_num_cols = test.select_dtypes('number').columns.tolist()
        numerical_variables = list(set(test_num_cols).intersection(set(numerical_variables)))
    
    result = get_numerical_data_summary(train, numerical_variables)
    if test is not None:
        test_result = get_numerical_data_summary(test, numerical_variables)
        return display('Train Data', result, 'Test', test_result)
    return display('Train', result)


## Data clustering

def get_optimal_clusters(df, n_clusters=13, scale=False, 
                         scorer:Literal['silhouette', 'elbow']='elbow'):
    """
    Gets the optimal number of clusters using the Silhouette or Elbow methods

    :param df: Input Data
    :param n_clusters: Number of clusters to check
    :param scale: Boolean. To standardize input data
    :param scorer: Method to use for selecting optimal clusters (silhouette or elbow)
    """
    if not isinstance(df, DataFrame):
        df = DataFrame(df)
    X = df.copy()
    scores = []
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    for i in tqdm(range(2, n_clusters+1)):
        res = KMeans(i, max_iter=500, n_init=10)
        res.fit(X)
        if scorer == 'silhouette':
            scores.append(silhouette_score(X, res.labels_))
        elif scorer == 'elbow':
            scores.append(res.inertia_)

    # visualise
    fig, ax = plt.subplots(1, figsize=(8,4.5))
    ax.plot(list(range(2,n_clusters+1)), scores, 'o-', linewidth=1.4, markersize=3)
    ax.set_xticks(range(2, n_clusters+1, 2), range(2, n_clusters+1, 2))
    ax.set_xlabel('Number of clusters')
    ax.set_title(f'{scorer.title()} Method', loc='left', fontweight='bold', fontsize=10)
    ax.set_ylabel(f'{scorer.title()} scores', fontweight='bold')
    fig.tight_layout()
    plt.show()


def cluster_data(df, n_cluster, scale=False):
    """
    Fits a KMeans clustering algorithm based on a defined number of clusers

    :param df: Pandas DataFrame
    :param n_cluster: Number of clusters to group data
    :param scale: Boolean. Standardize the input values

    :returns cluster labels and centers for clustered data
    """
    X = df.copy()

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_cluster, n_init=10, random_state=34, max_iter=500)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_