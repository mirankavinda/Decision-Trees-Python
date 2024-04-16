from sklearn.tree import DecisionTreeClassifier

# Sample data
X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

# Fit the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict
prediction = model.predict([[5]])
print(prediction)
