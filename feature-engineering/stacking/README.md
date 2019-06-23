# stacking

[Kaggle-Titanic-Kernel-Stacking](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/comments)

Now as alluded to above in the introductory section, stacking uses predictions of base classifiers as input for training to a second-level model. 

However one cannot simply train the base models on the full training data, generate predictions on the full test set and then output these for the second-level training. 

This runs the risk of your base model predictions already having "seen" the test set and therefore overfitting when feeding these predictions.