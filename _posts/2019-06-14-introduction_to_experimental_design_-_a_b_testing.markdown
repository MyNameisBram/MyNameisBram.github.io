---
layout: post
title:      "Introduction to Experimental Design - A/B Testing"
date:       2019-06-14 18:06:37 +0000
permalink:  introduction_to_experimental_design_-_a_b_testing
---


Companies large and small, continuously run experiments in order to stay competitive, attract new customers, retain current customers and last but not least, increase revenue. Data scientists are able to help in evaluating experiments and tests new or existing features, implement, and come to a conclusion on which features are better suited for the occasion and provide recommendations, in order to streamline decision making. 

A/B testing and hypothesis testing seems to be generalized as the same thing. It’s not quite the same. A/B testing is an experiment design, while hypothesis testing is a statistical technique for generating inferences (conclusion) from data.  

Good experimental design is needed to draw accurate conclusions from your experiments. The scientific method is used for experimental design because the process is designed in a way to help answer important questions with as little ambiguity as possible. The general structure of the method is: 1) Make an observation 2) Examine the research 3) Form a hypothesis 4) Conduct an experiment i.e. hypothesis testing 5) Analyze results, see if results are statistically significant 6) Draw a conclusion. It’s important to note, when forming the hypothesis, an educated guess of our outcome, we are defining our Alternative hypothesis, the one we want to prove correct (e.g. statement: discounts increase quantity ordered vs. null hypothesis, discounts have no effect on quantity ordered ), while the null hypothesis is the opposite, and we want to reject! (e.g. discounts have no effect on quantity ordered). 

Good experiments shows the independent variable (x) has an effect on the dependent variable (y) because we control other things that could affect (y), until a conclusion can be made that what happened to (y) is because of (x). An important aspect is having a control group, a cohort that does not receive treatment or feature tested (from our recent example, no discounts). So, the control group are customers that purchased without discount. Another important aspect is sample size. Small sample sizes are susceptible to randomness issues, where large sample sizes protects from randomness and variance. Reproducibility is very important in experimental design. This means that if Jane and John Doe follows the steps outlined, they should produce very similar results, allowing for randomness and natural variance. Here is a link to a detailed example of [hypothesis testing](https://github.com/MyNameisBram/Module-2-Project/blob/master/student.ipynb). 


What is A/B testing? According to [Optimizely](https://www.optimizely.com/optimization-glossary/ab-testing/), A/B testing is a method of comparing two versions of webpage or app against each other to determine superior performance. A/B testing is an experiment where two or more variants of a page are shown to users at random, and statistical analysis is used to determine which variation performs better, provided a goal. 

Why? A/B testing measures and collects each experience in order to analyze whether changing the experience had a positive, negative, or no effect on user behaviour. It can also be used consistently to improve a given experience, which down the road will likely lead to improvement in goals. For example, a B2B company desires to increase their sales lead quality and volume from campaign landing pages. A sample of A/B testing application might look like this, a team would change headlines, visuals, color, form fields, call to action, and overall layout of the page. Over time, the effect of multiple “success” changes from experiments can be combined to provide insight on the measurable improvement of the “new” vs old experience. 

The framework normally looks like this: 
- Collect data
- Identify goals   
- Generate Hypothesis
- Create variations
- Analyze results

During our analysis, there should be a statistically significant difference between the old and new version, in order to call it a success. Here is an example of [A/B testing using python](https://www.kaggle.com/tammyrotem/ab-tests-with-python)



