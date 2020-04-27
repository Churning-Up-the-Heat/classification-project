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
    ax = sns.lineplot(x="tenure", y="churn_encoded", data=df, color="Purple")
    plt.ylabel('Churn Rate')
    

def lineplot_rate_of_churn_to_tenure_years(df):
    '''function to plot rate of churn and tenure (in years)'''
    df_plot = df.groupby('tenure_years').churn_encoded.mean().reset_index()
    plt.figure('figure', figsize=(13, 10))
    plt.title("Tenure in years vs Rate of Churn", fontsize=17)
    ax = sns.lineplot(x="tenure_years", y="churn_encoded", data=df, color="Purple")
    plt.ylabel('Churn Rate')

    
def isolated_tenure_distros(train):
    '''Plots the disrobution of Tenure for all contract types individually'''
    month_to_month = train[train.contract_type == "Month-to-month"]
    one_year = train[train.contract_type == "One year"]
    two_year = train[train.contract_type == "Two year"]

    plt.figure(figsize=(13,10))
    plt.suptitle('Isolated Visualizations of Tenure Distributions - note Y axis is not on the same scale', fontsize=14)
    
    #Distro for All contract types
    plt.subplot(2,2,1)
    plt.title("All Contract Types", fontsize=14)
    sns.distplot(train.tenure, color="Purple")
    plt.ylabel("Count")
    
    #Distro for Month to Month
    plt.subplot(2,2,2)
    plt.title("Month to Month Contracts", fontsize=14)
    sns.distplot(month_to_month.tenure, color="Purple")

    #Distro for one year
    plt.subplot(2,2,3)
    plt.title("One Year Contracts", fontsize=14)
    sns.distplot(one_year.tenure, color="Purple")

    #Distro for two year
    plt.subplot(2,2,4)
    plt.title("Two Year Contracts", fontsize=14)
    sns.distplot(two_year.tenure, color="Purple")
    
    
def tenure_distros_overlayed(train):
    '''Plots the disrobution of Tenure for all contract types and overlays them for comparision'''
    month_to_month = train[train.contract_type == "Month-to-month"]
    one_year = train[train.contract_type == "One year"]
    two_year = train[train.contract_type == "Two year"]

    plt.figure(figsize=(13,10))
    plt.title("Tenure Distributions by Contract Type", fontsize=14)
    sns.distplot(month_to_month.tenure, color="Purple", label="Month to Month")
    sns.distplot(one_year.tenure, color="Grey", label="One Year")
    sns.distplot(two_year.tenure, color="Blue", label="Two Year")
    plt.legend()
    
    
    
def monthly_charges_distros(train):
    '''Plots the disrobution of Monthly Charges for all contract types'''
    month_to_month = train[train.contract_type == "Month-to-month"]
    one_year = train[train.contract_type == "One year"]
    two_year = train[train.contract_type == "Two year"]

    plt.figure(figsize=(15,12))
    plt.suptitle('Distribution of Monthly Charges - note Y axis is not on the same scale', fontsize=18)
    
    #Distro for All contract types
    plt.subplot(2,2,1)
    plt.title("All Contract Types", fontsize=14)
    sns.distplot(train.monthly_charges, color="Purple")
    plt.ylabel("Count")
    
    #Distro for Month to Month
    plt.subplot(2,2,2)
    plt.title("Month to Month Contracts", fontsize=14)
    sns.distplot(month_to_month.monthly_charges, color="Purple")

    #Distro for one year
    plt.subplot(2,2,3)
    plt.title("One Year Contracts", fontsize=14)
    sns.distplot(one_year.monthly_charges, color="Purple")

    #Distro for two year
    plt.subplot(2,2,4)
    plt.title("Two Year Contracts", fontsize=14)
    sns.distplot(two_year.monthly_charges, color="Purple")  

    
def churn_percentages_at_12_months(train):
    '''
    Gives us a stacked bar plot for each of the contract types churn rates when they reach the
    12 month marker
    '''
    df_tenure_at_one_year = train[train.tenure == 12]
    x1 = "contract_type"
    x2 = "churn"
    (df_tenure_at_one_year.groupby(x1)[x2]
     .apply(lambda s: s.value_counts(normalize=True)) # custom aggregation to get value counts by group
     .unstack() # turn an index into columns
     .plot.bar(stacked=True, width=.9, color=["Grey", "Purple"]))
    plt.title("Churn percentage for each contract type at 12 months")
    plt.legend(title=x2)
    plt.xticks(rotation=0)
    plt.xlabel('')    
        

def stacked_barplot_for_churn_rates_by_contract(train):
    '''
    Helps visualize the churn rates for each contract type in a stacked bar plot
    '''

    x1 = "contract_type"
    x2 = "churn"

    plt.rc('figure', figsize=(13, 10))
    plt.rc('font', size=13)

    (train.groupby(x1)[x2]
     .apply(lambda s: s.value_counts(normalize=True)) # custom aggregation to get value counts by group
     .unstack() # turn an index into columns
     .plot.bar(stacked=True, width=.9, color=["Grey", "Purple"]))
    plt.legend(title=x2)
    plt.xticks(rotation=0)
    plt.xlabel('')

    
def initial_corr_heatmap(df):
    '''heatmap to explore correlations of numerical categories'''
    plt.figure('figure', figsize=(13, 10))
    plt.title("Heatmap of Correlations", fontsize=13)
    sns.heatmap(df.corr(), cmap='Purples', fmt='.2%')


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

    #5. plot phone_lines
    plot_categorical_with_churn_rates(train, "phone_lines")
    
    #6. plot online_security
    plot_categorical_with_churn_rates(train, "online_security")
    
    #7. plot device_protection
    plot_categorical_with_churn_rates(train, "device_protection")
    
    #8. plot device_protection
    plot_categorical_with_churn_rates(train, "online_backup")
    
    #9. plot tech_support
    plot_categorical_with_churn_rates(train, "tech_support")
    
    #10. plot streaming_tv
    plot_categorical_with_churn_rates(train, "streaming_tv")
    
    #11. plot streaming_movies
    plot_categorical_with_churn_rates(train, "streaming_movies")
    
    #12. plot paperless_billing
    plot_categorical_with_churn_rates(train, "paperless_billing")
    
    #13. plot internet_service_type
    plot_categorical_with_churn_rates(train, "internet_service_type")
    
    #14. plot payment_type
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


def price_threshold_internet_services(train):
    '''Barplot allows us to look at price Thresholds fot the different internet service types'''
    train["monthly_bins"] = pd.cut(train.monthly_charges,3)
    plt.figure(figsize=(13, 10))
    sns.barplot(x="monthly_bins", y="churn_encoded", data=train, hue="internet_service_type", color="m")
    plt.hlines(y=train.churn_encoded.mean(), xmin=-1, xmax=3, ls=":")
    plt.title("Price Threshold for Internet Service Type")
    
    
def price_threshold_phone_lines(train):
    train["monthly_bins"] = pd.cut(train.monthly_charges,5)
    plt.figure(figsize=(13, 10))
    sns.barplot(x="monthly_bins", y="churn_encoded", data=train, hue="phone_lines", color="m")
    plt.hlines(y=train.churn_encoded.mean(), xmin=-1, xmax=5, ls=":")
    plt.title("Price Threshold for Phone Services where 0 is none, 1 is single line, 2 is multi line")