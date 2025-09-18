# **Predicting Medical Insurance Cost for an Insurance Company**

### **Project Goal:**
The goal of this project is to build a Machine Learning model that can predict medical expenses for customers for an Insurance company and provide insights into the major factors that contribute to higher insurance costs.


### **Problem Context:**
Insurance companies invests a lot of time, effort, and money in creating models that accurately predicts health care costs.
The purposes of this project is to look into different features to observe their relationship to predict individual medical costs billed by health insurance.

## **Notebook & Data**

 The Jupyter Notebook containing the analysis of the problem can be found here: [Notebook](https://github.com/sreela-gopi/Capstone_Assignment/blob/main/Capstone_Assignment_Initial_Report.ipynb)<br>
 The dataset provided for the analysis can be found here : [Dataset](https://github.com/sreela-gopi/Capstone_Assignment/blob/main/data/medical_insurance.csv)

## **Findings**

**Findings from Exploratory Data Analysis(EDA)**
1. **Age**: The distribution of age is quite varied.The average age of policy holders is 39 years. The data shows a wide range from a minimum of 18 to a maximum of 64 years.
2. **BMI**: The average BMI is around 30.6, which falls into the obese range (30.0 or higher) according to the WHO. The values range from 15.96 to 53.1
3. **Children**: The number of children per policyholder ranges from 0 to 5. The average number of children is 1 and a significant portion of the population has few to no children.
4. **Charges**: The average charges is 13,279.12. The standard deviation is very high, at 12,110.35 suggesting a large spread in costs. The minimum charge is 1,121.87 and the maximum is $63,770.42
5. **Sex**: The number of males and females is almost balanced, with 675 males and 662 females.
6. **Smoker**: The majority of the individuals are non-smokers (1063), while there are 274 smokers.
7. **Region**: The data is around evenly distributed across the four regions, with the southeast having the highest count at 364 individuals, followed by the southwest (325), northwest (324), and northeast (324).
8. **Charges vs. Age**: As people get older, their medical costs tend to increase.The data points appear to be grouped into distinct bands, which may suggest that other factors are having a significant impact at different age groups.
9. **Charges Vs. BMI**: Individuals with higher BMI values tend to have higher charges.
10. **Charges Vs. No. of Children**: There is no clear trend indicating that an increase in the number of children leads to a consistent increase in medical costs.
11. **Charges by Sex**: While charges are generally similar for both genders, some individuals face significantly higher costs.
12. **Charges by Smoker Status**: Smoking status is a major driver of charges.
13. **Charges by Region**: Charges are relatively similar across all four regions.

## **Actionable insights to the Business**
**Actionable Insights from Univariate Analysis:**<br>
1. **Charges**: The insurance company should focus on risk management for high-cost individuals. Since a small portion of the policyholders accounts for a disproportionately large share of the total medical costs, developing targeted programs for those groups could significantly reduce overall payout.
2. **Age**: The age distribution of policyholders is fairly even. So the compoany's marketing strategies must be appealing to all age groups.
3. **BMI**: The average BMI is approximately 30, which falls into the obese category. The data also has outliers with extremely high BMI values.
        This indicates a potential public health concern within the customer base. The insurance company could introduce wellness programs           or rewards to encourage healthy habits among policyholders.
4. **Smoker**: Around 20% of policyholders are smokers which could lead them to serious health issues. The company could design policies with higher premiums for smokers to accurately reflect their higher risk profile. Additionally, offering smoking cessation programs could be a strong value-add for customers.
5. **Region**: The customer base is fairly evenly distributed across the four geographical regions. There's no single dominant region. The company can develop region-specific strategies,for example, a region with a higher average BMI might be targeted with specific wellness programs.
6. **Children**: The majority of policyholders do not have children, while a smaller portion has one or two children. The maximum number of children is five. The company can develop personalized product recommendations based on a policyholder's family structure.
7. **Sex**: There is a nearly equal distribution of male and female policyholders, with males making up slightly more of the total population
            There could be some strategies developed if there is any difference in average charges among males and females.

**Actionable Insights from Bivariate Analysis:**<br>
1. **Impact of Smoker status on Charges**: Smoking status is a primary driver of their insurance costs. Insurance company could develop targeted wellness programs to encourage smoking cessation. Offering premium discounts or other rewards for non-smokers could also be a strategy to manage costs and promote healthier lifestyles among policyholders. This could lead to a potentially lower future claims.
2. **The Minimal Impact of Sex and Region**: There is no significant difference in medical charges between genders or across regions, these factors are not strong predictors of charges. So the insurance company could focus of other factors primarily.

   

## **Tech Stack**

**Python:** Pandas, NumPy, scikit-learn <br>
**Data Visualization:** Matplotlib, Seaborn

This assignment was completed as part of the Capstone Assignment of Professional Certificate in Machine Learning & Artificial Intelligence - 2025
