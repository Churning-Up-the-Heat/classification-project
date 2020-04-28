# Churning Up the Heat

# Project Goals
- Identify drivers behind churn rates for Telco customers

# Deliverables 
* Link to Presentation [here](https://docs.google.com/presentation/d/1_ox9z4ZivH6vF1zdNttJ3oWlPMXx75xsreFynchkl08/edit?usp=sharing)

* model.py - python file taking in data and generates a csv of predictions

* predictions.csv - Contains the results of running the model.py file

# Data Dictionary
|      DataFrame Column      |          SQL Column         |                                             Description                                            |
|:--------------------------:|:---------------------------:|:--------------------------------------------------------------------------------------------------:|
| tenure                     | tenure                      | How long (in months) a customer has been with the company                                          |
| contract_type_encoded      | contract_type               | Contract types: Monthly, 1 year, 2 years                                                           |
| monthly_charges            | monthly_charges             | What the customer pays monthly. Assessed to be the last recorded month's payment                   |
| senior_citizen             | senior_citizen              | Determines if the customer a senior citizen or not                                                 |
| payment_type_encoded       | payment_type                |  Payment types: Electronic Check, Mailed Check, Bank transfer (automatic), Credit card (automatic) |
| churn_encoded              | churn                       | Determines if a customer has discontinued their service for the company's product                  |
| phone_lines                | phone_service/multiple_line | Determine is a customer has phone service, and how many lines they have                            |
| dependent_partner_grouping | dependents/partner          | Determines if the costumer has a partner, dependents, or neither                                   |