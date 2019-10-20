# Grade_Predictor
Uses a machine learning library to train a bot to predict students' grades

System Requirements: Anaconda, Pandas, NumPy, scikit-learn

This program takes a record of students' academic performance from a tri-semester educational
institution and predicts students' 3rd semester grades based on their 1st semester grades, 
2nd semester grades, study time, instances of failures, and numbers of absences
Data Source: https://archive.ics.uci.edu/ml/datasets/student+performance

Assumption: There exists a strong correlation between the students' 3rd semester grades
and their 1st semester grades, 2nd semester grades, time spent studying, and number of times
they have failed or missed school. Therefore, we can use a linear regression model to train
our bot
