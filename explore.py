import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split


def lineplot_rate_of_churn_to_tenure_months(df):
    '''function to plot rate of churn and tenure (in months)'''
    df_plot = df.groupby('tenure').churn_encoded.mean().reset_index()
    plt.figure('figure', figsize=(13, 10))
    plt.title("Tenure in months vs Rate of Churn", fontsize=17)
    sns.lineplot(df_plot.tenure, df_plot.churn_encoded, color="Purple")
    plt.ylabel('Churn Rate')


def lineplot_rate_of_churn_to_tenure_years(df):
    '''function to plot rate of churn and tenure (in years)'''
    df_plot = df.groupby('tenure_years').churn_encoded.mean().reset_index()
    plt.figure('figure', figsize=(13, 10))
    plt.title("Tenure in years vs Rate of Churn", fontsize=17)
    sns.lineplot(df_plot.tenure_years, df_plot.churn_encoded, color="Purple")
    plt.ylabel('Churn Rate')


def corr_heatmap(df):
    '''heatmap to explore correlations of numerical categories'''
    plt.figure('figure', figsize=(13, 10))
    plt.title("Heatmap of Correlations", fontsize=13)
    sns.heatmap(df.corr(), annot=True, cmap='Purples', fmt='.2%')


def plot_categorical_with_churn_rates(df, column_name):
    '''plots bar chart for categorical feature vs tenure'''
    plt.figure(figsize=(10, 7))
    plt.title(f"Churn Rates by {column_name}")
    df.groupby(column_name).churn_encoded.mean().plot.bar(ec='black', fc='purple', width=.9, label='')
    plt.xticks(rotation=0)
    plt.xlabel('')
    plt.ylabel('Churn Rate')
    plt.hlines(df.churn_encoded.mean(), *plt.xlim(), ls='--', color='grey', label='average churn rate')
    plt.legend()


def plot_all_categoricals_with_churn_rates(train):
    '''plots bar chart for each categorical feature vs tenure'''
    #1. plot contract_type
    plot_categorical_with_churn_rates(train, "contract_type")
    
    #3. plot partner
    plot_categorical_with_churn_rates(train, "partner")

    #5. plot phone_service
    plot_categorical_with_churn_rates(train, "phone_service")
    
    #6. plot multiple_lines
    plot_categorical_with_churn_rates(train, "multiple_lines")
    
    #7. plot online_security
    plot_categorical_with_churn_rates(train, "online_security")
    
    #8. plot device_protection
    plot_categorical_with_churn_rates(train, "device_protection")
    
    #9. plot device_protection
    plot_categorical_with_churn_rates(train, "online_backup")
    
    #10. plot tech_support
    plot_categorical_with_churn_rates(train, "tech_support")
    
    #11. plot streaming_tv
    plot_categorical_with_churn_rates(train, "streaming_tv")
    
    #12. plot streaming_movies
    plot_categorical_with_churn_rates(train, "streaming_movies")
    
    #13. plot paperless_billing
    plot_categorical_with_churn_rates(train, "paperless_billing")
    
    #14. plot internet_service_type
    plot_categorical_with_churn_rates(train, "internet_service_type")
    
    #15. plot payment_type
    plot_categorical_with_churn_rates(train, "payment_type")
    
    
def churn_rate_for_contract_types_at_12_months(train):
    '''For every unique contract type it return the churn percentage within that group'''
    x1 = "contract_type"
    x2 = "churn"
    (train.groupby(x1)[x2]
     .apply(lambda s: s.value_counts(normalize=True)) # custom aggregation to get value counts by group
     .unstack() # turn an index into columns
     .plot.bar(stacked=True, width=.9, color="bg"))
    plt.title("Churn percentage for each contract type at 12 months")
    plt.legend(title=x2)
    plt.xticks(rotation=0)
    plt.xlabel('')
    
def stats_for_contract_types(train):
    '''
    Takes in a training dataframe and returns a new dataframe with the following:
    Min, max, std, median for each contract type
    '''
    
    monthly_charges_mean = pd.DataFrame((train.groupby("contract_type")).monthly_charges.mean())
    monthly_charges_mean.columns = ['Mean monthly charges']
    
    monthly_charges_median = pd.DataFrame((train.groupby("contract_type")).monthly_charges.median())
    monthly_charges_median.columns = ['Median monthly charges']
    
    monthly_charges_max = pd.DataFrame((train.groupby("contract_type")).monthly_charges.max())
    monthly_charges_max.columns = ['Max monthly charges']
    
    monthly_charges_min = pd.DataFrame((train.groupby("contract_type")).monthly_charges.min())
    monthly_charges_min.columns = ['Min monthly charges']
    
    monthly_charges_std = pd.DataFrame((train.groupby("contract_type")).monthly_charges.std())
    monthly_charges_std.columns = ['STD monthly charges']
    
    summary1 = pd.merge(monthly_charges_mean, monthly_charges_median, left_index=True, right_index=True)
    
    summary2 = pd.merge(monthly_charges_max , monthly_charges_min, left_index=True, right_index=True)
    
    summary3 = pd.merge(summary1 , summary2 , left_index=True, right_index=True)
    
    df = pd.merge(summary3 , monthly_charges_std, left_index=True, right_index=True)
    
    return df