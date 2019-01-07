from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

mnist = fetch_mldata('MNIST original')
# These are the images
# There are 70,000 images (28 by 28 images for a dimensionality of 784)
print(mnist.data.shape)
# These are the labels
print(mnist.target.shape)
print(mnist.DESCR)
#
# Split Data into Training and Test Sets (MNIST)
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1 / 7.0, random_state=0)
#
# Show the Images and Labels (MNIST) ... one set
plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(train_img[0:10], train_lbl[0:10])):
    plt.subplot(2, 5, index + 1)
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)
plt.show()
#
# Scikit-learn 4-Step Modeling Pattern (MNIST)
# step 1: Make an instance of the Model
# default solver is incredibly slow thats why we change it
'''
solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
The solver for weight optimization.

‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
‘sgd’ refers to stochastic gradient descent.
‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands
 of training samples or more) in terms of both training time and validation score. 
 For small datasets, however, ‘lbfgs’ can converge faster and perform better.
'''
logisticRegr = LogisticRegression(C=0.1, solver='lbfgs')  # score = 0..9135
# logisticRegr = MLPClassifier(solver = 'adam') # Multi-layer Perceptron  --- score = 0.964
#
# step 2 Training the model on the data, storing the information learned from the data
logisticRegr.fit(train_img, train_lbl)
#
# step 3. Predict labels for new data (new images)
# returns numpy array
# Predict for One Observation (image)
print(f'test_digit: {test_lbl[0]}, predicted_digit : {logisticRegr.predict(test_img[0].reshape(1, -1))}')
# Predict labels for new data (new images)
print(f'test_digits:  {test_lbl[1:10]}, predicted_digits : {logisticRegr.predict(test_img[1:10])}')
# Predict labels for entire dataset
predictions = logisticRegr.predict(test_img)
# Use score method to get accuracy of model
score = logisticRegr.score(test_img, test_lbl)
print(score)
#
# Display Misclassified images with Predicted Labels (MNIST)
index = 0
misclassifiedIndexes = []
for label, predict in zip(test_lbl, predictions):
    if label != predict:
        misclassifiedIndexes.append(index)
    index += 1

plt.figure(figsize=(20, 4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(test_img[badIndex], (28, 28)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], test_lbl[badIndex]), fontsize = 15)
plt.show()
