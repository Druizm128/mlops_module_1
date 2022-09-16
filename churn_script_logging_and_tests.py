import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='logs/test_results.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = cl.import_data(r"./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_data: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err
        
def test_eda():
    '''
    test perform eda function
    '''
    try:
        df = cl.import_data(r"./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        # Create dependent variable
        df['Churn'] = (df['Attrition_Flag']
            .apply(lambda val: 0 if val == "Existing Customer" else 1))
        cl.perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except:
        logging.info("Testing perform_eda: The EDA has some problem")
        

def test_encoder_helper():
    '''
    test encoder helper
    '''
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
    try:
        df = cl.import_data(r"./data/bank_data.csv")
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(df)
        assert set(X_train.columns).intersection(cat_columns) == len(cat_columns)
        assert set(X_test.columns).intersection(cat_columns) == len(cat_columns)
        assert set(X_train.columns).intersection(quant_columns) == len(quant_columns)
        assert set(X_test.columns).intersection(quant_columns) == len(quant_columns)

        X_train_clean, X_test_clean = cl.encoder_helper(X_train,
                                                        X_test,
                                                        category_lst=cat_columns,
                                                        quant_lst=quant_columns)

        assert X_train_clean.columns == X_test_clean.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except ModuleNotFoundError as err:
        logging.info("Testing encoder_helper: FAILURE")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''
    
if __name__ == "__main__":
    test_import()