from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics


digits = load_digits()

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print('Image Data Shape', digits.data.shape)
# Print to show there are 1797 labels (integers from 0–9)
print("Label Data Shape", digits.target.shape)
print(f'DESCR: {digits.DESCR}')
#
#
plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(digits.data[0:10], digits.target[0:10])):
    plt.subplot(2, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)
plt.show()
#
# split Data into Training and Test Sets (Digits Dataset)
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
#
# Scikit-learn steps for Modeling Pattern (Digits Dataset)
#
# step 1: Make an instance of the Model
logisticRegr = LogisticRegression()
#
# step 2 Training the model on the data, storing the information learned from the data
logisticRegr.fit(x_train, y_train)
#
# step 3. Predict labels for new data (new images)
# returns numpy array
# Predict for One Observation (image)
print(f'test_digit: {y_test[0]}, predicted_digit : {logisticRegr.predict(x_test[0].reshape(1,-1))}')
# Predict labels for new data (new images)
print(f'test_digits:  {y_test[0:10]}, predicted_digits : {logisticRegr.predict(x_test[0:10])}')
# Predict labels for entire dataset
predictions = logisticRegr.predict(x_test)
# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)
'''
Confusion Matrix (Digits Dataset)
A confusion matrix is a table that is often used to describe the performance of a classification model 
(or “classifier”) on a set of test data for which the true values are known. 
'''
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
plt.show()
