                            Task
                        Spam Email Detection

Objective:
    To classify the messages either a spam on non spam, by training the machine with the given dataset of the sample messages provided with the labels either as 'ham' or 'spam' based on their characteristics classification should be made possible.

Models used:
1. Support Vector Machine
2. K Neighbor Classifier 
3. Mutinomial Naive Bayes
4. Decision Tree Classifier
5. Linear Regession Technique
6. Random Forest Classifier
7. AdaBoost Classifier
8. Bagging Classifier
9. Extra Tree Classifier

Prerequisites:
For building a model to fulfill our aim we need following:
 1. Input File
  The given test dataset with the messages and their identified labels as either ham or spam
 2. Set of Libraries
  The model will reqire variety of set of libraries, some will be required to be installed previouslyin the the notebook prior to the import statements. Installation can be by the command:
        pip install module_name      or
        pip install --upgrade module_name          // for the updated versions of modules

Steps performed:
Building the required model has required following steps to be performed:

1. Installing and imorting the necessary libraries
    With the necessary installation step we have importthe following modules:
    a)pandas for the dataframes conversions, reading- writing data
    b)matplolib
    c)seaborn
    d)scikitlearn
    e)nltk
    f)warnings and os module are supplemantary


2. Sourcing the input file
    With pandas we can read the files from various formats into the dataframe . In this we have used excel format insttead can also use csv format too. This input file will help our program train the model about type of messages based on the text characteristics. We have used this file:
     File : Spam Email Detection.xlsx
    This dataframe will help us to process and obtain the desired functions and results.

3. Data cleaning
    In this step we have checked for the inconsistencies in the given dataset from the above input file viz., a) describing its statstical measures
    b) format of the data(head())
    c) columns names 
    d) shape of the dataset (i.e. rows , columns), 
    e) checked for any null values and their removal, 
    f) looked for duplicate values and removal
    g) dropping unnecessary 
    h) renaming columns or adding numeric labels into the dataframe

4. Model BUilding
  a) initialize the models :
     That means calling the constructors of the respective classes of the various models we've  used so that their intialization could happen which will then help to tarin, classify the data.

  b) split the data into training and test sets : 
    From the given input dataset / created dataframe we will divide the data into two sets for theb test and training purpose. The dataframe will be divided into the feature_test, feature_train and labels_test, labels_train . The tests sets here will be used to verify the classifications done by the models do matches the exact label in the dataset or datframe i.e. the feature_train will contain messages while label train their respective labels  either ham or spam  which will train our program hoe to classsfy a text, then model on its own will revoke messages from features_test and identify their labels_test.

  c) train the models :
     Tp train the models we use fit() method that actually tarin any model with the given dataset. We've used nine models to train in the for loop to avoid repeatition of the statement with the help of dictionary.

5. Evaluate the models
 i) Find the scores
    Once a prediction is done we will find scores of each classifier for four of the following measures:
    a) Accuracy :
       It is defined as the number of correct predictions divided by the total number of predictions multiplied by 100.  It calculates the ratio of correctly predicted instances to the total instances.

    b) Precision :
       Precision provides the accuracy of the positive prediction made by the classifier. The equation is as follows:
                Precision = True Positive / (True Positive + False Positive)

    c) Recall score :
       It is ratio of positive instances that are truly detected by the model.

    d) F1 score :
       The F1 score is the mean of precision and recall. It favors classifiers that have similar precision and recall.

ii) Visualise the scores :
 We've used visualisation for the simplicity of understanding and better analysis of the metric. One is heatmap another is discrete color bar graphs for each of the modeling techniques we've used.

 iii) Confusion matrix:
      This is a table i. e., used to describe the performance of a classification model. It presents a overview of the predictions made by the model against the actual class labels. The confusion matrix is a matrix with 4 combinations of predicted and actual classes: 
      True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN).

6. Classify the message :
   With one of the model techniques above we will classify the text that user will enter as either spam or ham. For this we've used Multinomial Naive Bayes Classifier as it is widely used for the text classificaton.
