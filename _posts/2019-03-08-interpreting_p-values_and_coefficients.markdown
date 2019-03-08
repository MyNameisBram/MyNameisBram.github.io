---
layout: post
title:      "Interpreting P-values and Coefficients "
date:       2019-03-08 16:36:58 +0000
permalink:  interpreting_p-values_and_coefficients
---

## Interpreting P-values in Linear Regression
P-value represents the probability that the coefficient is actually zero. A low p-value or less than .05, means you reject your null hypothesis and generally, accept the alternative hypothesis.. A p-value greater than .05, indicates accepting the null hypothesis and therefore rejecting the alternative hypothesis. Why is p-value important? I understood it like this, A p-value lower than .05 in the independent variable (predictor or x-axis) is likely a significant addition to the model because changes in the predictor’s (independent variable or x-axis or input) value are related or impacts changes in the features. (dependent variable or y-axis or response or outcome).

Values are considered statistically “significant’ when p-value is below .05. Conversely, a p-value larger than .05 is considered insignificant because changes in the predictor does not related to changes in the feature.

In our current project, we are working with King County House Sales dataset to determine which predictors impact home price. The data set provided 19 independent variables. After visually exploring our data using `seaborn.jointplot()` we noticed linear relationships in most of the predictors, of which, the top three includes: sqft_living, grade, and sqft_above. 

After running our data in the test models using `import statsmodels.formula.api as smf`, we found further evidence in backing our previous statement. The p-values for the top three predictors were under .05. We can be confidently state, that the p-value of our top three predictors are significant. 

## Interpreting Coefficients in Linear Regression 
Regression is inherently a model about the outcome variable. The coefficient represents the **mean change in the outcome/response variable**, for one unit of change in the predictor variable while keeping other predictors constant/unchanged. Coefficients are easier to understand when visualized as **slopes**. Coefficients describe the relationship between a predictor and the response. It also specify the direction of the relationship between a predictor and a response. When both increases, a positive sign is reflected. When a predictor increases and the response decrease, a negative sign is reflected (think top left corner to bottom right corner). 

The coefficient indicates that for every additional x, you can expect an increase in y, by an average of the mean. If the line is flat, i.e. a slope coefficient of zero, the response would not change even if the predictor increases in value. Therefore, a low p-value indicates that the slope is not zero, which means that change in the predictor and change in the response are linked to some degree. 

Looking to our King County project, the predictor sqft_living, has a coefficient of 0.30. This coefficient represents the mean increase of price for every additional square feet in living. If your sqft_living increase by 1000, the average price increase by 30%. Suppose the regression line of price vs. sqft_living was flat, which means a slope coefficient of zero. In this scenario, the mean price would not change regardless how far along the line moves. A coefficient near zero, suggests that predictor used, has no effect on response and you’ll see a high p-value (insignificant) corresponding  to the near zero coefficient. 


#### Side note: JupyterLab  *(off topic but useful) *
As an aspiring Data Scientist, I’ve noticed and learned that  you spend most of your time on data wrangling and data exploratory. I would say between 50-80% of your time and energy is exhausted towards this part of the process. 

A few tools I’ve used and learned so far are; Terminal, Python,  Anaconda-Navigator, Jupyter Notebooks/Labs, pandas, matplotlib, NumPy, and Seaborn. 

One of the most useful tools I’ve come across in my Data Science bootcamp so far, is Jupyter Labs. The Jupyter Labs interface provides building blocks for interactive, exploratory computing.  Jupyter Labs can do everything Jupyter Notebook can do and more. The Jupyter Notebook has grown many of its components like, terminal, file browser, and text editor. Jupyter Lab brings all those components together in a single unifying platform to create a better workflow. (insert picture example of how you can split the screen). 

One of the main reason I like using Juypter Labs vs. Notebook, is the interactive features. It allows you to have multiple tabs (think browser tabs), work with a Notebook side by side, drag and drop cells which creates an easy way to organize your notebook. It’s also nice because we use terminal locally and interact with github to clone, commit and git push files without leaving the interface. In conclusion, accessibility and utility is the reason Jupyter Labs is my go to tool for documentation and creating a notebook. 

