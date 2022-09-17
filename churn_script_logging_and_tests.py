import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='logs/test_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True)


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cl.import_data(r"./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error(f"Testing import_data: FAILED")
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            """Testing import_data: The file doesn't appear to have rows and columns""")


def test_eda():
    '''
    test perform eda function
    '''
    try:
        df = cl.import_data(r"./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        # Create dependent variable
        df['Churn'] = (
            df['Attrition_Flag'] .apply(
                lambda val: 0 if val == "Existing Customer" else 1))
        #######
        cl.perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except (FileNotFoundError, AttributeError) as err:
        print(err)
        logging.error(f"""Testing perform_eda: There was a problem building the
        graphs. Either the path or a variable was not found.""")


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    try:
        df = cl.import_data(r"./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        # Create dependent variable
        df['Churn'] = (
            df['Attrition_Flag'] .apply(
                lambda val: 0 if val == "Existing Customer" else 1))

        ########
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
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
            df, quant_columns, cat_columns)
        assert len(set(X_train.columns).intersection(
            cat_columns)) == len(cat_columns)
        assert len(set(X_test.columns).intersection(
            cat_columns)) == len(cat_columns)
        assert len(set(X_train.columns).intersection(
            quant_columns)) == len(quant_columns)
        assert len(set(X_test.columns).intersection(
            quant_columns)) == len(quant_columns)
        logging.info("Testing test_perform_feature_engineering: SUCCESS")
    except (AssertionError) as err:
        print(err)
        logging.error("Testing encoder_helper: FAILURE")


def test_encoder_helper():
    '''
    test encoder helper
    '''
    try:
        df = cl.import_data(r"./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        # Create dependent variable
        df['Churn'] = (
            df['Attrition_Flag'] .apply(
                lambda val: 0 if val == "Existing Customer" else 1))

        ########
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
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
            df, quant_columns, cat_columns)
        assert len(set(X_train.columns).intersection(
            cat_columns)) == len(cat_columns)
        assert len(set(X_test.columns).intersection(
            cat_columns)) == len(cat_columns)
        assert len(set(X_train.columns).intersection(
            quant_columns)) == len(quant_columns)
        assert len(set(X_test.columns).intersection(
            quant_columns)) == len(quant_columns)

    #########
        X_train_clean, X_test_clean = cl.encoder_helper(
            X_train, X_test, category_lst=cat_columns, quant_lst=quant_columns)

        assert len(X_train_clean.columns) == len(X_test_clean.columns)
        logging.info("Testing test_encoder_helper: SUCCESS")
    except (AssertionError) as err:
        print(err)
        logging.error("""Testing encoder_helper: FAILURE the categorical values
        are different between train and test.
        """)


def test_train_models():
    '''test train_models'''
    try:
        df = cl.import_data(r"./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        # Create dependent variable
        df['Churn'] = (
            df['Attrition_Flag'] .apply(
                lambda val: 0 if val == "Existing Customer" else 1))

        ########
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
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
            df, quant_columns, cat_columns)
        assert len(set(X_train.columns).intersection(
            cat_columns)) == len(cat_columns)
        assert len(set(X_test.columns).intersection(
            cat_columns)) == len(cat_columns)
        assert len(set(X_train.columns).intersection(
            quant_columns)) == len(quant_columns)
        assert len(set(X_test.columns).intersection(
            quant_columns)) == len(quant_columns)

        X_train_clean, X_test_clean = cl.encoder_helper(
            X_train, X_test, category_lst=cat_columns, quant_lst=quant_columns)

        #########
        logging.info("Modelling ...")
        cl.train_models(X_train_clean, X_test_clean, y_train, y_test)
        logging.info("Testing test_encoder_helper: SUCCESS")
    except (ValueError) as err:
        print(err)
        logging.error("""Testing encoder_helper: FAILURE there is a wrong hyper
        parameter.
        """)


if __name__ == "__main__":
    test_import()
    test_eda()
    test_perform_feature_engineering()
    test_encoder_helper()
    test_train_models()
