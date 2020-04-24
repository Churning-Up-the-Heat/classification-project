import pandas as pd

from sklearn.linear_model import LogisticRegression

def choose_features(train, validate):
    X_train = train[['tenure', 
                     'contract_type_encoded', 
                     'monthly_charges', 
                     'payment_type_encoded']]
    y_train = train.churn_encoded

    X_validate = validate[['tenure', 
                           'contract_type_encoded', 
                           'monthly_charges', 
                           'payment_type_encoded']]
    y_validate = validate.churn_encoded 
    
    return X_train, y_train, X_validate, y_validate
        
def create_log_reg_model(train, validate):
    # Split by Features to be Used
    X_train, y_train, X_validate, y_validate = choose_features(train, validate)           
    
    # Create and Fit the Model
    log_reg_model = LogisticRegression().fit(X_train, y_train)
    
    # Predict
    predictions = pd.DataFrame(
        {'actual': y_validate,
         'log_reg_model_predictions': log_reg_model.predict(X_validate),
         'log_reg_model_probabilities': log_reg_model.predict_proba(X_validate)[:, 1]
        })

    # Generate csv
    predictions.to_csv('predictions.csv')
    
    return log_reg_model, predictions
    
    
    