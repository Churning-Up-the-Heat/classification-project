# Why are our customers Churning?
To answer this question we had to figure out the reasons behind churning. We need to understand which of out customers are churning and of those customers what is driving them to churn. We have done an in depth exploratory analysis to help answer these questions and create a model that will enable us evaluate future churn.

# Environment Setup:
- ENV.py file with the following information as it pertains to the SQL network:
    - password
    - username
    - host
- Python install through anaconda

# Project Goals

1. Report with detailed analysis in .ipynb format
2. CSV file containing customer_id, probability of churn, and the prediction of churn (1=churn, 0=not_churn)
3. Google Slides explaining model chosen and brief analysis for the Senior Leader Team (SLT)
4. All files necessary to recreate our findings and models
5. GitHub repo containing all files

# Deliverables 
* Link to Presentation [here](https://docs.google.com/presentation/d/1_ox9z4ZivH6vF1zdNttJ3oWlPMXx75xsreFynchkl08/edit?usp=sharing)

* model.py - python file taking in data and generates a csv of predictions

* predictions.csv - Contains the results of running the model.py file

# Key Questions to Answer 
1. Are there clear groupings where a customer is more likely to churn? What if you consider contract type? Is there a tenure that month-to-month customers are most likely to churn? 1-year contract customers? 2-year customers? Do you have any thoughts on what could be going on? (Be sure to state these thoughts not as facts but as untested hypotheses. Unless you test them!). Plot the rate of churn on a line chart where x is the tenure and y is the rate of churn (customers churned/total customers).

2. Are there features that indicate a higher propensity to churn? like type of Internet service, type of phone service, online security and backup, senior citizens, paying more than x% of customers with the same services, etc.?

3. Is there a price threshold for specific services where the likelihood of churn increases once price for those services goes past that point? If so, what is that point for what service(s)?

4. If we looked at churn rate for month-to-month customers after the 12th month and that of 1-year contract customers after the 12th month, are those rates comparable?

# Tested Hypothesis
- $H_0$ - Being a senior citizen is independent of churning
- $H_0$ - Gender is independent of churning
- $H_0$ - Having dependents is independent of churning


# Data Dictionary
| DataFrame Column           | SQL Column                  | Description                                                                                        |
|----------------------------|-----------------------------|----------------------------------------------------------------------------------------------------|
| tenure                     | tenure                      | How long (in months) a customer has been with the company                                          |
| contract_type_encoded      | contract_type               | Contract types: Monthly, 1 year, 2 years                                                           |
| monthly_charges            | monthly_charges             | What the customer pays monthly. Assessed to be the last recorded month's payment                   |
| senior_citizen             | senior_citizen              | Determines if the customer a senior citizen or not                                                 |
| payment_type_encoded       | payment_type                |  Payment types: Electronic Check, Mailed Check, Bank transfer (automatic), Credit card (automatic) |
| churn_encoded              | churn                       | Determines if a customer has discontinued their service for the company's product                  |
| phone_lines                | phone_service/multiple_line | Determine is a customer has phone service, and how many lines they have                            |
| dependent_partner_grouping | dependents/partner          | Determines if the costumer has a partner, dependents, or neither                                   |

# Project Conclusions

- Final model able to predict churn 81% of the time

- Senior Citizens are very likely to churn

- Customers on the Month-To-Month contract have the highest churn rates

- Customers who have Fiber Optics are also likely to churn - this is surprising since it is a premium service

- Customers who have dependents are less likely to churn


## Key Takeaways
    - What kind of contract the customer has plays a large role in if they will churn:
        * ~43% of customers on monthly contracts are likely to churn, as opposed to yearly (~12%) and two years(~3%)
    - The most churn occurs in the first few months of tenure, and then again at 6 years of tenure (~70 months)
        * This is either an indication that customers are churning for some reason before getting a 4th two year contract, or the data we are working with is only looking at a period of time about 6 years long
        
    - Senior citizens are more likely to churn:
        * Making up 17 percent of the customer base, being a senior citizen does not impact modeling
        * Senior citizens pay more on monthly rates. Offering lower monthly rates could retain more senior citizens


## Moving Forward
- Gather more information as to why customers are choosing the month-to-month contract. By doing this we should be able to optimize our model in detecting churn. 

- Do a cross market analysis to see if we can determine why our customers with fiber optics are churning, we can solve questions like are our rates to high for this particular service?

- We found that approximately 41% of our Senior Citizens are churning and their rates on average are 17 dollars higher. We could run a promotion specifically for this demographic to help reduce their churn rate.

# Technical Skills
- Python (including internal and third party libraries)
- SQL
- Hypothesis testing
- Logistic Regression, Decision Tree, Random Forest and K Nearest Neighbor

# Data Source for project:
Link may be found [HERE](https://ds.codeup.com/7-classification/project/)

