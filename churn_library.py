'''
This package includes all the functions to train 
and evaluate a customer churn prediction problem

Author: Dante Ruiz
'''
# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report
from ast import main
import os
import logging
os.environ['QT_QPA_PLATFORM']='offscreen'
sns.set()

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
    
    logging.info("Performing EDA ...")
    print(df.head())
    print(df.shape)
    print(df.isnull().sum())
    print(df.describe())

    plt.figure(figsize=(20,10)) 
    df['Churn'].hist();

    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist();

    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar');

    plt.figure(figsize=(20,10)) 
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True);

    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.show()

def encoder_helper(train_X, test_X, category_lst, quant_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with X_train_clean
            df: pandas dataframe with X_test_clean
    '''
    # One Hot Encoding
    logging.info("One hot encoding categorical variables ...")
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(train_X.loc[:, category_lst])
    ohe_labels = ohe.get_feature_names_out()
    X_train_cat_clean = pd.DataFrame(ohe.transform(train_X.loc[:, category_lst]).toarray(), columns = ohe_labels)
    X_test_cat_clean = pd.DataFrame(ohe.transform(test_X.loc[:, category_lst]).toarray(), columns = ohe_labels)
    # Cleaning train-test
    X_train_quant = train_X.loc[:, quant_lst]
    X_test_quant = test_X.loc[:, quant_lst]
    X_train_clean = pd.concat([X_train_quant.reset_index(), X_train_cat_clean.reset_index()], axis = 1)
    X_test_clean = pd.concat([X_test_quant.reset_index(), X_test_cat_clean.reset_index()], axis = 1)
    return (X_train_clean, X_test_clean)


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
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
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

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
    pass


if __name__ == "__main__":
    
    logging.info("Executing program ...")
    # Import data
    df = import_data(r"./data/bank_data.csv")
    # Create dependent variable
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1) 
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
    # Create X and Y
    y = df['Churn']
    X = df.loc[:, quant_columns + cat_columns]
    # Train Test Split
    logging.info("Splitting train-test ...") 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)    

    # One Hot Encoder
    logging.info("Preprocessing ...") 
    X_train_clean, X_test_clean = encoder_helper(X_train, X_test, category_lst=cat_columns, quant_lst=quant_columns)

    logging.info("Training models ...")
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Grid search
    rfc = RandomForestClassifier(random_state=42)
    param_grid = { 
        'n_estimators': [200],
        #'n_estimators': [200, 500],
        #'max_features': ['auto', 'sqrt'],
        #'max_depth' : [4,5,100],
        #'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    logging.info("Training Random Forest ...")
    cv_rfc.fit(X_train_clean, y_train)
    logging.info("Training Logistic Regression ...")
    lrc.fit(X_train_clean, y_train)

    logging.info("Predicting ...")
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train_clean)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test_clean)

    y_train_preds_lr = lrc.predict(X_train_clean)
    y_test_preds_lr = lrc.predict(X_test_clean)

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
    # plots
    logging.info("Generating ROC curves ...")
    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test_clean, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, X_test_clean, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.show()
    # Save best models
    logging.info("Saving models ...")
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    # Load best models
    logging.info("Loading models ...")
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')
    # Plot from best models
    logging.info("Generating ROC curves ...")
    lrc_plot = RocCurveDisplay.from_estimator(lr_model, X_test_clean, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(rfc_model, X_test_clean, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.show()
    # Shap values
    logging.info("Generating Shap Value Plot for Random Forest ...")
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test_clean)
    shap.summary_plot(shap_values, X_test_clean, plot_type="bar")
    # Calculate feature importances
    importances = cv_rfc.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_train_clean.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_train_clean.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_train_clean.shape[1]), names, rotation=90);
    plt.rc('figure', figsize=(5, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off');
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off');
    logging.info("Execution SUCCESSFULL")