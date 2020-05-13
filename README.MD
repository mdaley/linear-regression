# Machine Learning Stuff

Based on the linear regression task from the Coursera Machine Learning course. I wanted
to see whether I could implement machine learning algorithms in C++.

Technologies used here:

- C++ 11 (could be a higher version but sticking to this until I need higher)./
- Eigen - C++ matrix library.
- dlib - an alternative C++ matrix library, with added machine learning algorithms.
- matplotlib - a Python graphing library.
- matplotlib_cpp - a C++ wrapper for matploglib.
- Python 3.

## Exercise 1a - simple linear regression

Given X and Y points from input data generate a line of best fit using linear regression.

Original points with line of best fit generated using linear regression:

![linear regression chart](https://github.com/mdaley/ml/raw/master/images/ex1a-1.png)

Cost against values of Theta:

![linear regression chart2](https://github.com/mdaley/ml/raw/master/images/ex1a-2.png)

## Exercise 1b - linear regression with multiple variables

Graph showing how learning rates differ based on different _alpha_ increments during
the learning process:

![learinng rates chart](https://github.com/mdaley/ml/raw/master/images/ex1b.png)

## Exercise 2a - simple logistic regression

Given data that includes two exam results plus whether or not the candidate was admitted into
the college setting the exams, work out how to predict a candidates likelihood of success based
on a pair of results. Plot a line showing the 0.5 boundary between admission and non-admission.

![logistic_regression](https://github.com/mdaley/ml/raw/master/images/ex2a.png)

## Exercise 2b - multi-variable logistic regression

Perform logistic regression using more many more variables, allowing the generation of much
more complex boundaries between two conditions. In the graph below the boundaries marks the point
at which a microchip has a 50% change of being successfully accepted. Different values of the 
_lambda_ correction factor alter the extent to which the results _fit_ the data. Lambda = 0 has a messy,
complex boundary, lambda = 1 seems more reasonable, lambda = 100 seems to be rather offset. (the boundaries
are made using scatter plots as the graphing library can't mathematically plot such complex shapes).

![logistic_regression2](https://github.com/mdaley/ml/raw/master/images/ex2b.png)