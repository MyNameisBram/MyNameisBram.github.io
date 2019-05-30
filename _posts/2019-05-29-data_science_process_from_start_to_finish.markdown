---
layout: post
title:      "Data Science process from Start to Finish"
date:       2019-05-30 01:51:03 +0000
permalink:  data_science_process_from_start_to_finish
---


When I started my data science journey, I was in for a rude awakening. I dove neck deep into heavy statistics, learning new computing languages, and detailed data science process. One of the things I wished I learned at the very beginning is, a combination of the Data science process using a dataset example. That is exactly what I plan on doing in this blog. I’ll be providing what a general learner data science notebook will look like with examples. The dataset I’m using is the Northwind database. Which is a free, open-source dataset created by Microsoft containing data from a fictional company.  

It’s good practice to know your which method, workflow, and/or process you plan on using.Today, I’ll be using a combination of  the OSEMiN method which stands for, Obtain, Scrub, Explore, Model, and iNterpret, and the Scientific method (like in grade school, testing your hypothesis, ect.). The OSEMiN method is one to the most common/popular method used in the data science workflow. 

Both the OSEMN and Scientific method is used for the framework of this project. Starting with retrieving and cleaning our data to avoid errors down the process. Then, exploring the data to get a "feel" of what we're looking at intuitively, using visualizations and calling methods to provide insight about our data. Many iterations will be made in going back and forth between wrangling and exploring our data to prepare the data for modeling. After our data is polished and clean, we proceed to feeding it into our model/algorithm. We then build and tweak models accordingly, in order to get the best results. Lastly, we interpret the results the best possible in order to either answer the questions provided or further the analysis is done, to provide a conclusion and or insight about new findings. 

#### Steps:

1. Define question 
2. Exploratory research 
    - Explore data 
3. Define hypothesis 
    - Define null and alternative hypothesis
    - Choose between using a one-tailed test (directional) or a two-tailed test (non-directional)
    - Set significance level $\alpha$ (alpha) - commonly set at 0.05
4. Statistical tests
    - Rationalize the appropriate statistical test
    - Checking assumptions 
    - Calculating test statistic and p-value 
    - Calculating effect size 
5. Conclusion(s)
    - Interpret outcome

#### Obtaining our Data

Below we are importing our necessary

```
#importing our libraries 

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline 

#importing sqlite database and checking table names

import sqlite3 as sq 
connection = sq.connect('Northwind_small.sqlite')
cur = connection.cursor()

#checking our table names
cur.execute('''SELECT name FROM sqlite_master WHERE type = 'table';''').fetchall()

```

```
[('Employee',),
 ('Category',),
 ('Customer',),
 ('Shipper',),
 ('Supplier',),
 ('Order',),
 ('Product',),
 ('OrderDetail',),
 ('CustomerCustomerDemo',),
 ('CustomerDemographic',),
 ('Region',),
 ('Territory',),
```


**Question:** Do discounts have a statistically significant effect on the number of products customers order? If so, at what level(s) of discount? 

**Null Hypothesis**: Discounts **does not** effect the number of products customers order.  
**Alternative Hypothesis**: Discounts **does** effect the number of products customers order. 

After obtaining our data, I separated the data into two dataframes, performed analysis of the number of  products purchased with/without discount and visualized the sum and mean of the data. I visually checked for normality using Q-Q plot and statistically, using SciPy's normaltest. After concluding that our data is not normal, I will use tests that performs well with non normal data. Parametric tests will be performed because parametric tests can perform well with continuous data that aren't normally distributed, **if** you satisfy the sample size in the guidelines [here](https://blog.minitab.com/blog/adventures-in-statistics-2/choosing-between-a-nonparametric-test-and-a-parametric-test) (1-sample t test, Greater than 20, ect.). I used the Welch's t-test because it performs well when assumption of normality isn't strong. 

```
# Selecting OrderDetail Table 
cur.execute('''SELECT * FROM orderdetail;''')
df_o_d = pd.DataFrame(cur.fetchall()) # creating pandas DataFrame
df_o_d.columns = [i [0] for i in cur.description] # label columns 


# Creating 2 different datasets, with and without discount
df_discount = df_o_d[df_o_d['Discount'] > 0 ]
df_no_discount = df_o_d[df_o_d['Discount'] == 0]

# A quick look at total and average quantity ordered
print("Total quantity ordered:",df_o_d['Quantity'].sum())
print("Total quantity ordered w/o Discount:",df_no_discount['Quantity'].sum())
print("Total quantity ordered w/ Discount:",df_discount['Quantity'].sum())

# print("Average quantity ordered per Product w/o Discount:",df_no_discount.groupby('ProductId')['Quantity'].mean())
# print("Average quantity ordered per Product w/ Discount:",df_discount['Quantity'].sum())

print("Average quantity ordered w/o Discount:", round(df_no_discount['Quantity'].mean(),2))
print("Average quantity ordered w/ Discount:", round(df_discount['Quantity'].mean(),2))
```

```
Total quantity ordered: 51317
Total quantity ordered w/o Discount: 28599
Total quantity ordered w/ Discount: 22718
Average quantity ordered w/o Discount: 21.72
Average quantity ordered w/ Discount: 27.11

```

Here we'll use **visualization** to provide an intuitive outlook on our analysis.

```
# Grouped by sum 
df_discount_sum = df_o_d[df_o_d['Discount'] > 0 ].groupby('ProductId')['Quantity'].sum()
df_no_discount_sum = df_o_d[df_o_d['Discount'] == 0].groupby('ProductId')['Quantity'].sum()

plt.figure(figsize=(16,5))
plt.bar(df_discount_sum.index, df_discount_sum.values, alpha=1, label='Discount', color='yellow')
plt.bar(df_no_discount_sum.index, df_no_discount_sum.values, alpha=0.8, label='No Discount', color='blue')
plt.legend()
plt.title('Order Quantity with/without discount')
plt.xlabel('Product ID')
plt.ylabel('Total Quantity', fontsize=18)
plt.show()

# Grouped by mean 
df_discount_mean = df_o_d[df_o_d['Discount'] > 0 ].groupby('ProductId')['Quantity'].mean()
df_no_discount_mean = df_o_d[df_o_d['Discount'] == 0].groupby('ProductId')['Quantity'].mean()

plt.figure(figsize=(16,5))
plt.bar(df_discount_mean.index, df_discount_mean.values, alpha=1, label='Discount', color='yellow')
plt.bar(df_no_discount_mean.index, df_no_discount_mean.values, alpha=0.8, label='No Discount', color='blue')
plt.legend()
plt.title('Order Quantity with/without discount')
plt.xlabel('Product ID')
plt.ylabel('Mean Quantity', fontsize=18)
plt.show()
```

```

```

```
print("Avg. quantity with Discount:",round(df_discount_mean.values.mean(),2))
print("Avg. quantity w/o Discount:",round(df_no_discount_mean.values.mean(),2))
```
```
Avg. quantity with Discount: 26.43
Avg. quantity w/o Discount: 21.81
```
Based on initial exploratory data analysis, orders with discount, has higher quantity average per order versus without discount. An average quantity of 4 more per order.

**Q-Q Plot** (Quantile - Quantile) is used to visually check for normality. **Q-Q Plot** (Quantile - Quantile) is used to visually check for normality. It’s important that we get our dataset as normal as possible to avoid problems down the process. After the Q-Q plot, we use statistical testing to double check for normality, using scipy’s normal-test. 

```
import numpy as np 
import statsmodels.api as sm
import pylab 

test_no = df_no_discount["Discount"]
test = df_discount["Discount"]
sm.qqplot(test, line='45')
pylab.title("With Discount")

sm.qqplot(test_no, line='45')
pylab.title("Without Discount")

pylab.show()
```

```

```

```
from scipy.stats import normaltest
normaltest(df_discount["Discount"], nan_policy='omit') # checking for normalization 
```

```
NormaltestResult(statistic=2005.0894275024564, pvalue=0.0)
```
**Interpretation:** 
The result of the p-value <= $\alpha$ of 0.05, we can reject the H$_0$  or null hypothesis, therefore the distribution is not normal, which leads me to use The Welch's t-test below. 

**Welch's t test** 
We saw earlier that our data is  abnormal. Here we are using The Welch's t-test vs. The Student t-test, because The student test assumes our data is normally distributed with samples having equal variance and sample size. The Welch's t-test is used when assumption of normalization don't hold strong. **Cohen's _d_:**
Here we are using the Cohen's _d_, one of the most common ways to measure effect size, to represent the difference between two or more groups. In which, larger values represent greater differentiation between the two groups. 

```
# Function for Cohen's d 
def Cohen_d(group1, group2): 
    diff = group1.mean() - group2.mean()
    n1, n2, = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()
    
    # Calculate the pooled threshold 
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic 
    d = diff / np.sqrt(pooled_var)
    return abs(round(d,4))
```

```
from scipy import stats # for significance levels, normality 

alt = df_o_d[df_o_d['Discount'] > 0 ]["Quantity"]
null= df_o_d[df_o_d['Discount'] == 0]["Quantity"]

# Using welch_t Test
t_stat, p = stats.ttest_ind(null, alt)
d = Cohen_d(alt, null)

print("Cohen's d:", d)
print("p-value:", p)
```

```
Cohen's d: 0.2863
p-value: 1.1440924523215966e-10
```

#### Observations:
The test results showed that discount has a statistical significance on the number of products customers order. A p-value of less than 0.05 declares some form of statistical significance therefore rejecting our null hypothesis. Cohen's _d_ level of .28 which is defined as **"small" effect size**. 


**P-Value =** probability sample Means are the same 

**Confidence Level** or (1 - P) **=** probability sample Means are different 

**Effect Size =** how different sample means are 

Why Effect Size matters? 
In data analytics domain, effect size calculation serves three primary goals:

* Communicate **practical significance** of results. An effect might be statistically significant, but does it matter in practical scenarios ?

* Effect size calculation and interpretation allows you to draw **Meta-Analytical** conclusions. This allows you to group together a number of existing studies, calculate the meta-analytic effect size and get the best estimate of the true effect size of the population. 

* Perform **Power Analysis** , which help determine the number of participants (sample size) that a study would require to achieve a certain probability of finding a true effect - if there is one. 

**Interpretation:** 
In the above case, the average quantity order with/without discount is **26.43(discount) - 21.81(no discount)** = difference of **4.62**. Since there is no **standard scale** to measure this difference, Cohen's d provide a way to define, **How big of a difference is this?** As described in the observation, the result shows a cohen's d level of .28 which equals to having a **small effect/significance**.

Below I'll use the same approach as above on the statistical significance _(if any)_ on discount. 

```
discounts = df_discount['Discount'].unique()
discounts.sort()

# converting to a DataFrame
groups = {}
for i in discounts:
    groups[i] = df_discount[df_discount['Discount']==i]
    
# labelling columns    
discounts_df = pd.DataFrame(columns=['Discount %','Orders','Avg. Order Quantity'])
for i in groups.keys():
    discounts_df = discounts_df.append({'Discount %':i*100,'Orders':len(groups[i]),'Avg. Order Quantity':groups[i]['Quantity'].mean()}, ignore_index=True)

discounts_df
```

```

Discount %	Orders	Avg. Order Quantity
0	1.0	1.0	2.000000
1	2.0	2.0	2.000000
2	3.0	3.0	1.666667
3	4.0	1.0	1.000000
4	5.0	185.0	28.010811
5	6.0	1.0	2.000000
6	10.0	173.0	25.236994
7	15.0	157.0	28.382166
8	20.0	161.0	27.024845
9	25.0	154.0	28.240260
```

We are removing discounts with an average quantity less than or equal to 2.0 because we feel like there isn’t much significance in keeping it. 

```
discounts_significance_df = pd.DataFrame(columns=['Discount %','p-value','Cohens d'], index=None)

discounts = [ 0.05, 0.1, 0.15, 0.2, 0.25]
control = df_o_d[df_o_d['Discount']==0]['Quantity']
for i in discounts:
    experimental = df_o_d[df_o_d['Discount']==i]['Quantity']
    st, p = stats.ttest_ind(control, experimental)
    d = Cohen_d(experimental, control)
    discounts_significance_df = discounts_significance_df.append( { 'Discount %' : str(i*100)+'%', 'p-value': p, 'Cohens d' : d } , ignore_index=True)    

discounts_significance_df
```

```

Discount %	p-value	Cohens d
0	5.0%	0.000011	0.3469
1	10.0%	0.015501	0.1959
2	15.0%	0.000011	0.3724
3	20.0%	0.000326	0.3007
4	25.0%	0.000018	0.3666
```

#### Observations: 
The test results showed that discounts above has a **statistical significance** on the number of products customers order vs. orders without discount. A p-value of less than 0.05 declares some form of statistical significance therefore **rejecting our null hypothesis**. Discounts 5%, 15%, 20%, 25% has a Cohen's d level of roughly 0.34 which is in between **small/medium effect size**. Discount of 10% has a Cohen's d level of 0.20 which is defined as small **effect size**. 

#### Interpretation:
We can also conclude, that discount rate of 15% and 25% shows it has the largest effect size of the discounts offered. We can check with the average quantity ordered, with 15% and 25% discount average order quantity of 28.

Above is a glimpse of what a data scientist notebook might look like. The dataset and tools used might be different, but the process and workflow will generally be inline to what you see above. Thanks for reading! 
