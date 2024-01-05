# Beat Care AI
A project report submitted in partial fulfilment of the
requirement for

## Disruptive Technologies- 1

(2 3 ECH-102)
By Group No - 02

S.No. UID Name Responsibility
1 23BAI70 043 Ramneek Kaur EDA of dataset
2 23BAI70 054 Ridhi PPT
3 23BAI70 059 Subhradip Majumder AI designing and implementation
4 23BAI70 069 Dipti Sinha Project Report
5 23BAI70 079 Rishabh Rao PPT


under the guidance of
Dr. Divneet Kapoor

## UIE, Chandigarh University


## Table of Contents

- List of Figures..........................................................................
- List of Tables...........................................................................
- Abstract.................................................................................
- Chapter 1. Introduction...............................................................
- Chapter 2. Background...............................................................
- Chapter 3. Proposed Framework.....................................................
- Chapter 4. Results......................................................................
- Chapter 5. Conclusion and Future scope...........................................
- References...............................................................................


## List of Figures..........................................................................

Figure 1.1: Flow diagram explanation

Figure 1.2: Distribution plots

Figure 1.3: Heat map

Figure 1.4: Confusion matrix/error plot

Figure 1.5: Compare models

Figure 1.6: Relation within columns

Figure 1.7: Website in picture

## List of Tables...........................................................................

Table 2.1: Algorithm used explanation

Table 2.2: Original dataset

Table 2.3: Dataset after EDA


## Abstract.................................................................................

About data science

Data Science is a multidisciplinary field that uses scientific methods, processes, algorithms,
and systems to extract insights and knowledge from structured and unstructured data. Data
scientists utilize various tools and techniques such as machine learning, data mining, and
visualization to derive actionable insights, solve problems, and inform strategic decision-
making in diverse industries, ranging from business and healthcare to finance and technology.

About problem

Since a lot of people are dying due to heart attack these days so it is important to know how
many beats are left for us. We felt that there is a need of a platform where the user can know
about their heart conditions from the comfort of their home. There should be a platform which
should tell the user about the chances of his survival the next year once they have encountered
a heart attack.

About proposed Solution

To design a website which will tell about the survival chances of a person based on their heart
records. The website will ask few basic questions from the user along with required data of
their heart like epss, lvdd etc. On the basis of this data our website will predict the chances of
survival of a person. It will also show the user about the accuracy of our prediction.

About Results

It will tell the person about his chances of survival in the upcoming time once they have already
encountered a heart attack. The user needs to enter some required details for the assessment
and the website will formulate the result on that basis. Since it works on binary classification
so it tells us two outputs – the person is good to go or the person is at high risk. In case of high
risk, the person can consult the doctor on time and take the necessary actions accordingly.


# 1. Chapter 1

## Introduction

## About Data science

Data Science is a multidisciplinary field that uses scientific methods, processes, algorithms,
and systems to extract insights and knowledge from structured and unstructured data. It
involves skills from statistics, mathematics, and computer science to analyse and interpret
complex datasets, uncover patterns, and make informed decisions. Data scientists utilize
various tools and techniques such as machine learning, data mining, and visualization to derive
actionable insights, solve problems, and inform strategic decision-making in diverse industries,
ranging from business and healthcare to finance and technology. The goal is to turn raw data
into meaningful information that can drive innovation and solve real-world challenges.

## Problem Statement

Since a lot of people are dying due to heart attack these days so it is important to know how
many beats are left for us. We felt that there is a need of a platform where the user can know
about their heart conditions from the comfort of their home.

Since, in the United States every year about 805,000 people have a heart attack. This is about
one heart attack every 40 seconds. Of these 605,000 are first heart attacks. So, the website
should tell the user about the chances of his survival the next year once they have encountered
a heart attack.

## Objectives

To design a website which will tell about the survival chances of a person based on their heart
records. We choose website because in current times the website is the more reliable than the
software itself because it requires less resources and time to respond, the only drawback is that
it requires a stable internet connection. The website will ask few basic questions from the user
along with required data of their heart like epss, lvdd etc. On the basis of this data our website
will predict the chances of survival of a person. It will also show the user about the accuracy
of our prediction.


# 2. Chapter 2

## Background

## Data set explanation

Data for classifying if patients will survive for at least one year after a heart attack.

Dataset has 133 entries and 13 instances

Certainly, let us break down the attributes of the dataset:

1. Survival (survival): This variable represents the number of months a patient has survived
(or is still surviving) since their heart attack. The variation in survival times is likely due to
patients having heart attacks at different times.
2. Still Alive (still-alive): A binary variable where 0 indicates that the patient did not survive
until the end of the survival period, and 1 means the patient is still alive. This variable helps
distinguish between patients who have survived less than a year and those who are still alive
after the specified period.
3. Age at Heart Attack (age-at-heart-attack): The age of the patient when they experienced a
heart attack. This variable provides information about the age at which heart issues occurred.
4. Pericardial Effusion (pericardial-effusion): A binary variable indicating the presence (1)
or absence (0) of fluid around the heart, known as pericardial effusion.
5. Fractional Shortening (fractional-shortening): A measure of heart contractility. Lower
numbers suggest increasingly abnormal contractions.
6. EPSS (epss): E-point septal separation, another measure of heart contractility. Larger
numbers indicate increasingly abnormal contractions.
7. LVDD (lvdd): Left ventricular end-diastolic dimension. It measures the size of the heart at
end-diastole, with larger hearts often associated with cardiac issues.
8. Wall Motion Score (wall-motion-score): This variable quantifies how segments of the left
ventricle are moving, providing insight into heart function.
9. Wall Motion Index (wall-motion-index): Calculated by dividing the wall motion score by
the number of segments seen in an echocardiogram. It is a normalized measure, usually based
on 12-13 segments seen in an echocardiogram.


10. Mult (mult): A derivative variable that can be ignored, likely not contributing directly to
the analysis or prediction tasks.
11. Name (name): This variable contains the names of the patients, which have been replaced
with placeholders like "name" to anonymize the dataset.
12. Group (group): A meaningless variable that can be ignored, as it does not provide relevant
information for the analysis.
13. Alive at 1 (alive-at-1): A Boolean variable derived from the first two attributes. A value of
0 indicates that the patient was either dead after 1 year or had been followed for less than 1
year. A value of 1 means the patient was alive at 1 year. This variable can be crucial for defining
the target variable in a binary classification task predicting whether a patient survives beyond
1 year.

Original dataset:


Dataset after EDA:

Algorithms used on data set

```
Literature Dataset Used Algorithm Used Accuracy
In our exploration of
text classification
methodologies, we
referenced seminal
works in the field of
Healthcare. Key
literature includes
studies showcasing
the effectiveness of
ML algorithms like
Logistic Regression
and the analysis of
proper data
preprocessing for
accurate model
performance.
```
```
Echocardiogram Logistic regression 0.
```

# 3. Chapter 3

## Proposed Framework

## Flow diagram explanation


Pseudo code

This Pseudocode demonstrates the implementation of the beat care ai which we developed in
this project.

INPUT: we are taking a few heart conditions which are required by the prediction model to
predict the value from the model itself. According to our data set we are taking 11 different
types of data as input and predicting a value as 0 and 1 accordingly.

OUTPUT: 1.00 prediction score.

Step 1 - Load python modules like pycaret, sklearn, joblib, pandas, matplotlib and django(for
frontend).

Step 2- Importing the dataset using the pandas library.

Step 3 - Then we perform a proper EDA or extensive data analysis.

Step 4- Now generating and refining the dataset more and more for further processes.

Step 5- Next we used the setup function to create a setup using pycaret library.

Step 6- Now to compared the model with compare model by using compare_model function
using pycaret.

Step 7- Where get to know our best pression model and the time it requires to predict it.

Step 8- Now save the model using joblib.

Step 9 - Load the model in djnago frame work and use it efficiently.


# 4. Chapter 4

## Results

## Distribution Plots

Bar graph -: Inference

1. The age around 20-40 have high chances of survival.
2. The age around 60- 80 have low chances of survival

## Heat Map


Feature Significance/importance

'age', 'pericardialeffusion', 'fractionalshortening', 'epss', 'lvdd', 'wallmotion-score'

These are the columns or instances which we are including to train our model and dropping the
rest since they are not of much importance and we are doing this using SimpleImputer method
for the same.

Confusion Matrix/Error Plot

The true prediction by model that the person will die is 42 and the true prediction that the
person will survive is 17

This confusion Matrix has plotted the relation between true and prediction values of the
dataset.


Compare Models

We compared all the models where we came to know that the best model is the Logistic regression
model which clearly showing an accuracy of 1.00, we choose Lr instead of the other models due to its
less time complexity.

Relation within columns

1. The relationship of columns with themselves have relationship of 1 (linear).
2. lvdd and epss have linear relationship. lvdd and fractional shortening have indirect relationship.

3. Survival rate is maximum when the age is around 30-40. Survival rate is maximum when the epss is
less.


# 5. Chapter 5

## Conclusion and Future Scope

To know the beat of a person once they have already suffered from the heart attack. It will tell
the person about his chances of survival in the upcoming time. We felt that there is a need of a
platform where the user can know about their heart conditions from the comfort of their home.
We choose that platform to be a website because we believe that its better an app in many ways
for example in android you cannot download many applications. We believe that this platform
should be accessible to everyone.

To build a website which will tell the user about their heart conditions. The website will ask
few basic questions from the user along with required data of their heart like epss, lvdd etc. On
the basis of this data our website will predict the chances of survival of a person. It will also
show the user about the accuracy of our prediction. Through this website they can know if their
heart conditions are good or not and if not then they can detect it and take required actions
accordingly.

Life is precious to all of us. We all want to stay close to our loved once and want to spend more
time with them. But the tragedy with life is that anything can change in a second. The base of
a human life is the heart so it needs to be taken care of as early as possible. As the old saying
goes “prevention is better than cure”, early detection of any heart issue can play a vital role in
saving our lives. This small step of detecting the problem at earliest and take necessary actions
towards it can help us to live more with our loved once.

Since we only live once so it is important to do it right. It is important to take care of our beats
with beat care.

We can see our Beat Care AI being used in the near future in:

- Real Time Heart Assessment Device: We can implement this Ai in the pacemaker (a
    device that continuously) read the heart beats and will predict the future possible heart
    diseases or problems which will be trained by our model using the data collected by the
    device.
- Genetically Inherited Heart Disease: Inherited heart conditions are caused by a
    fault (or mutation) in one or more of our genes. If after we can use certain device to
    monitor their heart then we can predict the future malfunctioning of patient’s heart.
- Sports: We can track and monitor the heartbeat of the player to predict the blood
    flow and the current condition of the body, it can also help the people with heart disease
    to get into sports since both doctor and the patient can perform and asses the patient’s
    heart real time during any sports.


Github Repository Link

https://github.com/subhradip32/Heart_Beat_ai

## Website in picture


## References...............................................................................

- UC Irvine Machine Learning Repository:
    https://archive.ics.uci.edu/dataset/38/echocardiogram
- https://pycaret.org/Classification/
- https://colab.research.google.com/notebooks/charts.ipynb#scrollTo=ZdEG-d4g4U6v


