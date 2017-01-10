# Iris

Analysing the hello world of Machine Learning, the iris dataset. 
There are 4 features namely:

1) Sepal Length
2) Sepal Width
3) Petal Length
4) Petal Width

The data set size is a meagre 150 samples of which, 120 are used for training and 30 samples are used as a validation set to test out predictions.

10-fold cross validation is used to reiterate through the training and avoid any overfitting.

Importing the sklearn library which is used for training and score calculation.
The dataset is taken from the UCI dataset repository.
The dataset is split into training data and the labels for training as well as testing data.
We analyse the features by plotting some basic graphs and to figure out the distribution these features belong to.
6 Models are used for training and their score are calculated.
The models include:
- Logistic Regression
- Linear Discriminant Analysis
- K Nearest Neighbour
- Decision Tree
- Naive Bayes
- Support Vector Machine

The scores of each algorithm are produced.

---------------------------------------------------------------------------------------------------------

Seperate implementation of KNN and SVM (Algorithms giving the best accuracy) added.

