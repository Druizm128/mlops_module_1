'''
This package includes all the functions to train
and evaluate a customer churn prediction problem
Author: Dante Ruiz
'''
# import libraries
import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, classification_report
import shap
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# Configure plot theme
sns.set()
# Constants
PATH_DATA = "./data"
PATH_EDA_IMAGES = "./images/eda"
PATH_RESULTS_IMAGES = "./images/results"
PATH_MODELS = "./models"
# Setup logging
logging.basicConfig(
    filename='logs/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    '''
    logging.info("Loading file ...")
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
        df: pandas dataframe

    output:
        None
    '''
    # EDA on data frame
    logging.info("Performing EDA ...")
    print(df.head())
    print(df.shape)
    print(df.isnull().sum())
    print(df.describe())
    # Class balance
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title("Class balance distribution")
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.savefig(f"{PATH_EDA_IMAGES}/churn_distribution.png")
    #plt.show()
    # Age distribution
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title("Customer age distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.savefig(f"{PATH_EDA_IMAGES}/customer_age_distribution.png")
    #plt.show()
    # Marital status distribution
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title("Customer marital status class distribution")
    plt.ylabel("Count")
    plt.savefig(f"{PATH_EDA_IMAGES}/marital_status_distribution.png")
    #plt.show()
    # Total transactions count
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title("Total transcations count distribution")
    plt.xlabel("Total transactions count")
    plt.ylabel("Count")
    plt.savefig(f"{PATH_EDA_IMAGES}/total_transaction_distribution.png")
    # Features corretlation
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Feature correlations")
    plt.savefig(f"{PATH_EDA_IMAGES}/heatmap.png")
    #plt.show()


def encoder_helper(train_X, test_X, category_lst, quant_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used
                  for naming variables or index y column]

    output:
        df: pandas dataframe with X_train_clean
        df: pandas dataframe with X_test_clean
    '''
    # One Hot Encoding
    logging.info("One hot encoding categorical variables ...")
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(train_X.loc[:, category_lst])
    ohe_labels = ohe.get_feature_names_out()
    X_train_cat_clean = pd.DataFrame(
        ohe.transform(train_X.loc[:, category_lst]).toarray(),
        columns=ohe_labels)
    X_test_cat_clean = pd.DataFrame(
        ohe.transform(test_X.loc[:, category_lst]).toarray(),
        columns=ohe_labels)
    # Cleaning train-test
    X_train_quant = train_X.loc[:, quant_lst]
    X_test_quant = test_X.loc[:, quant_lst]
    X_train_clean = pd.concat([
        X_train_quant.reset_index(), X_train_cat_clean.reset_index()],
        axis=1)
    X_test_clean = pd.concat(
        [X_test_quant.reset_index(), X_test_cat_clean.reset_index()],
        axis=1)
    return (X_train_clean, X_test_clean)


def perform_feature_engineering(df, quant_columns, cat_columns):
    '''
    input:
        df: pandas dataframe
        response: string of response name [optional argument that could be used
                  for naming variables or index y column]

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    # Create X and Y
    logging.info("Separate X, y ...")
    y = df['Churn']
    X = df.loc[:, quant_columns + cat_columns]
    # Train-test split
    logging.info("Splitting train-test ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder

    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    output:
        None
    '''
    # scores
    logging.info("Evaluating Random Forest ...")
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))
    logging.info("Evaluating Logistic Regression ...")
    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))


def feature_importance_plot(model, X_data, output_pth=None):
    '''
    creates and stores the feature importances in pth

    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
        None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(f"{PATH_RESULTS_IMAGES}/feature_importances.png")
    #plt.show()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models

    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data

    output:
        None
    '''
    logging.info("Training models ...")
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    rfc = RandomForestClassifier(random_state=42)
    # Grid search
    param_grid = {
        'n_estimators': [200],
        # 'n_estimators': [200, 500],
        # 'max_features': ['auto', 'sqrt'],
        # 'max_depth' : [4,5,100],
        # 'criterion' :['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    logging.info("Training Random Forest ...")
    cv_rfc.fit(X_train, y_train)
    logging.info("Training Logistic Regression ...")
    lrc.fit(X_train, y_train)
    # Generate predictions
    logging.info("Predicting ...")
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    # Model evaluation
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    # plots
    logging.info("Generating ROC curves ...")
    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(cv_rfc.best_estimator_,
                                   X_test,
                                   y_test,
                                   ax=ax,
                                   alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.title("Models ROC Curves")
    plt.savefig(f"{PATH_RESULTS_IMAGES}/model_roc_curves.png")
    #plt.show()
    # Save best models
    logging.info("Saving models ...")
    joblib.dump(cv_rfc.best_estimator_, f"{PATH_MODELS}/rfc_model.pkl")
    joblib.dump(lrc, f"{PATH_MODELS}/logistic_model.pkl")
    # Load best models
    logging.info("Loading models ...")
    rfc_model = joblib.load(f"{PATH_MODELS}/rfc_model.pkl")
    #lr_model = joblib.load(f"{PATH_MODELS}//logistic_model.pkl")
    # Shap values
    logging.info("Generating Shap Value Plot for Random Forest ...")
    explainer = shap.TreeExplainer(rfc_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    # Feature importances
    feature_importance_plot(rfc_model, X_test, output_pth=None)
    logging.info("Execution SUCCESSFULL")


if __name__ == "__main__":
    logging.info("Executing program ...")
    # Import data
    df = import_data(f"{PATH_DATA}/bank_data.csv")
    # Create dependent variable
    df['Churn'] = (df['Attrition_Flag']
                   .apply(lambda val:
                          0 if val == "Existing Customer" else 1))
    # Perform EDA
    perform_eda(df)
    # Variable selection
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]
    # Train Test Split
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, quant_columns, cat_columns)
    # One Hot Encoder
    logging.info("Preprocessing ...")
    X_train_clean, X_test_clean = encoder_helper(X_train,
                                                 X_test,
                                                 category_lst=cat_columns,
                                                 quant_lst=quant_columns)
    # Train models
    logging.info("Modelling ...")
    train_models(X_train_clean, X_test_clean, y_train, y_test)
