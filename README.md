# best sample
Finding best sample index by following steps:<br />
    Randomly select original data as training samples to ratio. Leave the remaining data as the out-of-bag samples.<br />
    Run 10 epochs:<br />
    * Train a logistic regression model via the selected training samples.<br />
    * Make predictions based on the training samples. Since we have totally no idea about the labels, only training data is available to measure the performance.<br />
    * If the new auc improved than the previous one, record the sample indexes and the auc.<br />
    * Remove the misclassified ones in the training samples and take the same amount of data from oob samples. If the oob samples is insufficient, exit the current epoch.<br />
    Use the final indexes as the training data to next classifier.<br />
