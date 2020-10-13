# 2-level-regressor ensemble for predicting length of stay of patients of Myocardial Infraction
The project was developed as a part of the national-level data science challenge at Phillips HackAbout and judged in the top 5.
Proposed a 2-level combination of classifier and regression neural network for imbalanced datasets, wherein samples of certain class outnumber the sample of other class.

### The problem we faced: 
Since the dataset was imbalanced, our initial predictions for `length of stay` were highly biased towards the mean. 

### Motivation behind the approach 
To overcome the problem of the imbalanced dataset, we divided the dataset into two classes -  `Near mean` (majority class) and `Far mean` (Minority class). **We tried to take into account the known prior information about the skewed class distribution by the use of ensembles**. Since it's easier to determine the boundary between the majority and minority classes, we first identify the class to which a data-point belongs to. Since now we have knowledge about the location of the data point in the class distribution, we can now improve now perform regression on the data-point.

### Methodology
The first module is a classifier, which tries to identify to which class the data point belongs to (i.e. Near-mean or Far-mean). 

The second module of the ensemble is a set of regression neural networks that are trained on different classes of the dataset. Based on the class-prediction from the classifier, one of the regression neural networks is chosen from the set and used to make predict length of stay of myocardial infarction.  

### Results
We attained an accuracy score of `87%` on the predicted length of stay. 
