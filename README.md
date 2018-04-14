# 2-level-regressor
2-level regressor for imbalanced dataset
Project was developed as a part of national level data science challenge at Phillips Hackabout. Judged in the top 5.
Proposed a 2-level combination of classifier and regression neural network for imbalanced datasets, wherein samples of certain class outnumber the sample of other class. First classifier is a far-mean/near mean classifier according to which the next  regression neural network is selected. Various different Regression neural network are trained based on their distance from the predicted means and the sample is fed to one of them according to the result of the previous neural network.
