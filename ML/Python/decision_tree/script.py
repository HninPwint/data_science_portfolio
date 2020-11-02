import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from decision_tree import TreeNode

#======================== Load Data ==============================
breast_cancer_db = load_breast_cancer()
X, y = breast_cancer_db["data"], breast_cancer_db["target"]
attributes = breast_cancer_db["feature_names"]

main_dataframe = pd.DataFrame(X, columns = attributes)
main_dataframe['Target'] = y

# Check the types of the variable
main_dataframe.dtypes
print(main_dataframe.head())


def discretize_continuous_variables(df, attributes):
    new_attributes = []
    for a in attributes:
        new_a = "Quant4." + a
        df[new_a] = pd.qcut(df[a], q=4, labels=["q1", "q2", "q3", "q4"])
        new_attributes.append(new_a) 
    return new_attributes  

#====== Discretize the variable =======
new_attributes = discretize_continuous_variables(main_dataframe, attributes)
df_new = main_dataframe[new_attributes]
df_new.head()

out_errs = 0
out_accuracies = 0

data = main_dataframe[new_attributes]
target = main_dataframe["Target"]

##### Split train set 80% and test set 20 % of the sample ######
train_size = 0.8     
index = int(train_size * len(data))
np.random.seed(52)

train_set, test_set = data[:index], data[index:]
train_target, test_target = target[:index], target[index:]


#### Tree Building using Training Set ######
dt = TreeNode(min_sample_num=20)
dt.fit(train_set, train_target)

out_errs = 0
out_accuracies = 0

data = main_dataframe[new_attributes]
target = main_dataframe["Target"]

##### Split train set 80% and test set 20 % of the sample ######
train_size = 0.8     
index = int(train_size * len(data))
np.random.seed(52)

train_set, test_set = data[:index], data[index:]
train_target, test_target = target[:index], target[index:]


#### Tree Building using Training Set ######
dt = TreeNode(min_sample_num=20)
dt.fit(train_set, train_target)


#### Model Prediction and Evaluation #######
from sklearn import metrics
def predict_and_evaluate(dataset, target):

    correct_prediction_count = 0
    err_false_positive = 0
    err_false_negative = 0
    all_predicted_values = []
    
    for (i, ct), actual_target_value in zip(dataset.iterrows(), target):
        
        predit_value = dt.predict(ct)
        all_predicted_values.append(predit_value)
        if predit_value and not actual_target_value:
            err_false_positive += 1
        elif not predit_value and actual_target_value:
            err_false_negative += 1
        else:
            correct_prediction_count += 1
    
    # Calculate Accuracy
    # accuracy = (correct_prediction_count/len(test_target)) * 100
    accuracy = metrics.accuracy_score(target, all_predicted_values)

    return err_false_positive, err_false_negative, accuracy
    

#### Model Prediction and Evaluation of Validation/ Test Sample ####
in_err_false_positive, in_err_false_negative, in_accuracy = predict_and_evaluate (train_set, train_target )
in_errs = (in_err_false_negative + in_err_false_positive)/len(train_target)
print("In-Sample Performance")
print("In-sample Error Rate", in_errs)
print("In Sample Accuracy Rate", in_accuracy)
print("=================")


err_false_positive, err_false_negative, accuracy = predict_and_evaluate (test_set, test_target )
out_errs = (err_false_negative + err_false_positive)/len(test_target)
out_accuracies = accuracy

 #### Print out Error and Accuracy #######
print("Out-Sample Performance")
print("Out-Sample Error Rate", out_errs)
print("Out-Sample Accuracy Rate", out_accuracies)
print("=================")