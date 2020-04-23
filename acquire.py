import pandas as pd

def get_telco_data():
    # Create a SQL query to get the data from SQL
    query = '''
            SELECT *
            FROM customers
            JOIN contract_types USING(`contract_type_id`)
            JOIN internet_service_types USING(`internet_service_type_id`)
            JOIN payment_types USING(`payment_type_id`)
            '''
    
    # Get the url for the SQL database, letting us query the telco_churn database
    telco_url = f'mysql+pymysql://{user}:{password}@{host}/telco_churn'
        
    return pd.read_sql(query, telco_url)