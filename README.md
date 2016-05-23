# best sample
Finding best sample index by following steps:<br />
1. Randomly select original data as training samples to ratio. Leave the remaining data as the out-of-bag samples.<br />
2. Run 10 epochs:<br />
3.    Train a logistic regression model via the selected training samples.<br />
4.    Make predictions based on the training samples. Since we have totally no idea about the labels, only training data is available to measure the performance.<br />
5.    If the new auc improved than the previous one, record the sample indexes and the auc.<br />
6.    Remove the misclassified ones in the training samples and take the same amount of data from oob samples. If the oob samples is insufficient, exit the current epoch.<br />
7. Use the final indexes as the training data to next classifier.<br />
