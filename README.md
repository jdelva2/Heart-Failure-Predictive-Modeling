# [Heart Failure Predictive Modeling](https://jdelva2.github.io/Heart-Failure-Predictive-Modeling/)

## Purpose
The purpose of this project was to test multiple algorithms to create a predictive model that would best predict patient death during the follow up period after the patient was diagnosed.
Some of the algorithms include:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

## Correlatios with scatterplot matrix
  Plotting the scatterplot matrix we can see the correlation and distribution between two variables. We can see that some of our data is binary. We also see that we have a time variable which could be used for a Time Series. However, due to the scope of this project we will more focus on checking and analyzing the data for model evaluation.
  
![](https://raw.githubusercontent.com/jdelva2/Heart-Failure-Predictive-Modeling/main/Graphs%20%26%20Results/features_pairplot.png?raw=true)

## Evaluating models with different metrics.
  Here we can see some of the values for the 4 evaluation metrics used to see which model performed that best.

![](https://github.com/jdelva2/Heart-Failure-Predictive-Modeling/blob/main/Graphs%20&%20Results/model_eval_metrics.png?raw=true)

## AUROC Curves for the 4 chosen models.
  Plotting the 4 models we can see which performed the best in terms of maximizing the FPR vs TPR.
![](https://github.com/jdelva2/Heart-Failure-Predictive-Modeling/blob/main/Graphs%20&%20Results/model_AUROC_curves.png?raw=true)

## Conclusion
  From the observed results we can see that the Logistic Regression model had the best: Accuracy, F1, and AUROC curve. Random Forest came close at 2nd best. However there are some things to note on why this was. Usually, Random Forest models outpreform simple regression models like Logistic Regression at larger scales. It's important to note the size of my data was 300 entries. However the data was split into 2. The first half for training/validation. The 2nd half was used as "never before seen" data. This was to act like the model was tested during deployment or in real time. If we were to improve the Random Forest model, we could test it with a larger dataset. As for the DecisionTreeClassifier, we could perform feature selection to select the best binary features. With XGBoost models, we can perform hyper-parameter tuning to select best parameters before training the model.
