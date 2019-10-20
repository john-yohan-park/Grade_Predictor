# Students_Grade_Predictor
Summary: Uses a machine learning library to train a bot to predict students' grades. Written in Python

System Requirements: Anaconda, Pandas, NumPy, scikit-learn

Takes students' academic performance record from a tri-semester educational
institution and predicts students' 3rd semester grades based on their 1st & 2nd 
semester grades, time spent studying, instances of failures, and number of absences.
Data Source: https://archive.ics.uci.edu/ml/datasets/student+performance

Assumption: There exists a strong correlation between the students' 3rd semester grades
and their 1st & 2nd semester grades, time spent studying, and number of times
they have failed or missed school. Therefore, we can use a linear regression model to train
our bot that predicts their 3rd semester grades
