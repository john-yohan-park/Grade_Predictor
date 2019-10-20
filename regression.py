'''
John Park
Github: john-yohan-park

System Requirements: Anaconda, Pandas, NumPy, scikit-learn

Takes record of students' academic performance from a tri-semester high school
and predicts their 3rd semester grades based on their 1st & 2nd semester grades,
time spent studying, instances of failures, and number of absences
Data Source: https://archive.ics.uci.edu/ml/datasets/student+performance

Assumption: There exists a strong correlation between the students' 3rd semester grades
and their 1st & 2nd semester grades, time spent studying, and number of times
they have failed or missed school. Therefore, we can use a linear regression model to train
our machine learning bot to predict the students' 3rd semester grades
'''
import pandas as panda      # extract data
import numpy as nump        # organize data into an array
import sklearn              # machine learning library
from sklearn import linear_model    # linear regression model

# extract data
data = panda.read_csv('student-mat.csv', sep=';')
# trim data to its desired attributes
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]    # working with ints

# predicted variable
predict = 'G3'

# construct new data frames
x = nump.array(data.drop([predict], 1))
y = nump.array(data[predict])

# partition 10% of data as test samples so we could test accuracy of our model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# train model using linear regression
linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)        # train: produces line of best fit from train data
acc = linear.score(x_test, y_test)  # test accuracy

predictions = linear.predict(x_test)    # array of just predictions

print("Accuracy: ", '\t', '\t', acc)
print("Guesses", '\t', '\t', "Actual")
for i in range(len(predictions)):
    print(predictions[i], '\t', y_test[i])
