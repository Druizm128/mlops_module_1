# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
## Project Description

- The model is implemented using best practices for refactoring a Jupyter Notebook MVP into a python script. 
- The python script is refactored using best practices such as logging and testing.
- The model consumes data from a flat file, executes a exploratory data analysis, data cleaning, feature engineering, training, model evaluation and saves best models.
- The model artifacts such as exploratory data analysis, best models and logs are saved in the images, models and logs directories.

## Files and data description
Overview of the files and data present in the root directory. 

.
├── churn_notebook.ipynb # Contains the model MVP
├── churn_library.py     # Python library with the churn functions and pipeline.
├── churn_script_logging_and_tests.py # Python library to test every function in the churn_library.py
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Read this data
│   └── bank_data.csv
├── images               
│   ├── eda              # Store EDA results in png
│   └── results          # Store modelling results in png
├── logs                 # Store logs
└── models               # Store models
## Running Files

1. To execute the machine learning pipeline you first need to install a python environment and install the libraries.
You can use pyenv virtualenv. 

2. Install the package dependencies to ensure reproducibility. By running:

    pip install -r requirements_py3.10.txt

3. Then clone the repository

4. Then run the pipline
    
    ipython churn_library.py

5. (Optional) If you want to run the test execute

    ipython churn_script_logging_and_tests.py

    or

    pytest churn_script_logging_and_tests.py



