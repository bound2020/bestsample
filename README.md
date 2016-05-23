# best sample
Finding best sample index by following steps:
1. Randomly select original data as training samples to ratio. Leave the remaining data as the out-of-bag samples.
2. Run 10 epochs:
3.    Train a logistic regression model via the selected training samples.
4.    Make predictions based on the training samples. Since we have totally no idea about the labels, only training data is available to measure the performance.
5.    If the new auc improved than the previous one, record the sample indexes and the auc.
6.    Remove the misclassified ones in the training samples and take the same amount of data from oob samples. If the oob samples is insufficient, exit the current epoch.
7. Use the final indexes as the training data to next classifier.
