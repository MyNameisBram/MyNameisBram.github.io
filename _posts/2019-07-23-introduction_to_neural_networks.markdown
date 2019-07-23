---
layout: post
title:      "Introduction to Neural Networks "
date:       2019-07-23 17:22:17 +0000
permalink:  introduction_to_neural_networks
---


- What is deep neural network?
- How is a network trained? 
- What is stochastic gradient method? 

## What is a neural network? 
A [neural network](https://www.xenonstack.com/blog/artificial-neural-network-applications/) is a group of algorithms that certify the underlying relationship in a set of data similar to the human brain. The neural network helps to change the input so that the network gives the best result without redesigning the output procedure. A neural network, generally consists of three different layers. An input layer, hidden layer(s), and output layer. 

Let's start with an example to get an idea of what a neural network is. Imagine a city has 5 hot dog vendors. We would like to predict the sales amount for a vendor, given certain features. Let's say you they are several features to predict sales of each vendor. Location, pricing, and variety of offerings. 

Our feature, location. One of the things that affect sales is the volume of people passing by the vendor stall, as these are all potential customers. This is largely driven by location. Next, let’s look at our feature, pricing. How affordable are the hotdogs at the locations will affect sales as well. Last, let’s look at the variety of offerings.  When a hot dog vendor offers more types of hot dogs and toppings, this might be perceived as a “better” than other vendors, just because customers have more to pick from. On the other hand, pricing might also affect perceived quality. Customer might think higher price means better quality and vice versa. 

In a neural network, all features will be connected with all nodes in the hidden layer, and weights will be assigned to the arrows shown below. Networks like this are also referred to as densely connected neural networks. 

![title](https://images.app.goo.gl/H2YJNakCjwPd8hac6/image.jpg)

To implement a neural network, we need to feed it inputs $x_i$ and the outcome $y$, and the features in the middle i.e. hidden layer with nodes representing hidden units, will be figured out automatically within the network. 

![title](Module04/section40/dsc-04-40-02-introduction-to-neural-networks-online-ds-ft-021119/figures/First_network.jpg)

## The power of deep learning:
In our example above, we have 3 inputs, a hidden layer with 4 nodes, and 1 output layer. Networks come in all different shapes and sizes. This is only one example of what a deep learning is capable of. 
- We can add more features (nodes) in the input layer.
- We can add more nodes in the hidden layer.
- We can add more hidden layers, which turns a neural network into a “deep” neural network.
- We can have several nodes in our output layer. 

One of the benefits of deep learning is that its extremely powerful. Unlike other statistical and machine learning techniques, deep learning deals well with unstructured (images, audio files, text data, ect.) data. 

| x | y |
|---|---|
| features of a hot dog vendor  | sales |
| Pictures of cats vs dogs | cat or dog? |
| Pictures of presidents | which president is it? |
| Dutch text | English text |
| audio files | text |
|  ... | ... |  

**Types of Neural Networks:**
- Standard neural network 
- Convolutional neural network (input = images, video) 
- Recurrent neural networks (input = audio files, text, time series data) - order of data matters
- Generative adversarial networks

## Logistic regression as a neural network: 
In logistic regression models, predictor, $\hat y$, is somewhere between 0 and 1. In classical regression, the expression $\hat y = w^T x + b$, can be problematic because it does not ensure that our $\hat y$ will be between 0 and 1, and it could be much bigger or even negative. 

We need to transform $w^T x + b$. For this example, we denote $\hat y = \sigma(w^T x + b)$, where $z = w^T x + b$, then $ \hat y = \sigma(z)$. This sigmoid function is an activation function used in neural networks. The expression for a sigmoid given by $\sigma(z) = \displaystyle\frac{1}{1 + \exp(-z)}$, it’s clear that it will always be between 0 and 1. 

![title](figures/sigmoid_smaller.png)

The neural network can be represented like the below: 

![title](figures/log_reg.png)


## Defining the loss and cost function:
A loss function is used to measure the difference between predicted value $\hat y$, and the actual label $y$. In logistic regression, the loss function is $\mathcal{L}(\hat y, y) = - ( y \log (\hat y) + (1-y) \log(1-\hat y))$. This **loss function** expression is convex, which makes using gradient descent easier. The **cost function** takes the average loss over all the samples: $J(w,b) = \displaystyle\frac{1}{l}\displaystyle\sum^l_{i=1}\mathcal{L}(\hat y^{(i)}, y^{(i)})$

The purpose of training your logistic model is ***minimize** your **cost function**.

The step we explained above is called **forward propegation.** 

Our cost function is a convex shape, and the idea you’ll start with an initial value of $w$ and $b$, then take a step in the steepest direction downhill, as we know is, gradient descent. 

Observing $w$ and $b$ separately, our algorithm will update both respectively and repeatedly in each step. 

$w := w- \alpha\displaystyle \frac{dJ(w)}{dw}$ and
$b := b- \alpha\displaystyle \frac{dJ(b)}{db}$

Remember that $ \displaystyle \frac{dJ(w)}{dw}$ and $\displaystyle \frac{dJ(b)}{db}$ represent the *slope* of the function $J$ with respect to $w$ and $b$ respectively. $\alpha$ is denoted the *learning rate*. 

This is called backpropagation, where you take the derivatives to calculate the difference between desired and calculated outcome, and repeat until the lowest cost value is found. 



## A few takeaways for our introduction to neural network: 

- Powerful algorithms, and can be tweaked using variations of nodes, layers, activation functions, ect…
- Simple neural network consists of a single-hidden-layer, which have similar properties as logistic regression models. 
- Performs well when using unstructured data. 
- Types include: convolutional NN, recurrent NN, and generative adversarial NN.
- Use loss and cost functions to minimize the “loss”. 
- Backward and forward propagation are used to estimate “model weights”

The content of your blog post goes here.
