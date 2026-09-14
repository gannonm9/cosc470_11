"""
Decision Tree Lab - Iris Dataset
================================
Each row is a flower with measured features and a known species label.
This is our LABELED training data -> supervised learning.

Fill in the TODOs. Estimated time: 10-12 minutes.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import OrdinalEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# 1. THE LABELED DATA
# -----------------------------------------------------------------
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["species"] = iris.target_names[iris.target]

print("Training data (this is what makes it SUPERVISED learning:")
print("every row has an input AND the correct answer/label):\n")
print(data)

# -----------------------------------------------------------------
# 2. TODO #1: Separate inputs (features) from the label (target)
# -----------------------------------------------------------------
# The model needs to learn a function: features -> label
# X = the columns the model is allowed to look at
# y = the column it's trying to predict
X = data.drop(columns=["species"])   # TODO: is this right? check it.
y = data["species"]                  # TODO: is this right? check it.

# Decision trees in sklearn need numbers, not text, so we encode.
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# -----------------------------------------------------------------
# 3. TODO #2: Create and train ("fit") the model
# -----------------------------------------------------------------
# .fit(X, y) is the moment "learning" happens: the tree searches for
# the sequence of yes/no questions about X that best predicts y.
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_encoded, y)          # TODO: call fit with the right arguments

# -----------------------------------------------------------------
# 4. Look at what it learned
# -----------------------------------------------------------------
feature_names = list(X.columns)
print("\nLearned decision rules:\n")
print(export_text(clf, feature_names=feature_names))

plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=feature_names, class_names=clf.classes_,
          filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree: Iris Species")
plt.tight_layout()
plt.savefig("iris_tree.png", dpi=150)
print("\nSaved a picture of the tree to iris_tree.png")

# -----------------------------------------------------------------
# 5. TODO #3: Predict on a NEW flower the model has never seen
# -----------------------------------------------------------------
# This is the payoff of supervised learning: generalizing to new,
# unlabeled examples using the pattern learned from labeled ones.
new_flower = pd.DataFrame({
    "sepal length (cm)": [6],
    "sepal width (cm)": [3],
    "petal length (cm)": [5],
    "petal width (cm)": [2],
})
new_flower_encoded = encoder.transform(new_flower)
prediction = clf.predict(new_flower_encoded)   # TODO: predict on new_flower_encoded

print(f"\nNew flower: {new_flower.to_dict(orient='records')[0]}")
print(f"Prediction: species = {prediction[0]}")

# -----------------------------------------------------------------
# 6. STRETCH GOAL (if time remains)
# -----------------------------------------------------------------
# Try changing max_depth to 1, then to 10. Re-run and compare the
# tree diagrams and the training accuracy. What happens? Why?
print("\nTraining accuracy by tree depth:")
for depth in [1, 3, 10]:
    depth_clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    depth_clf.fit(X_encoded, y)
    accuracy = depth_clf.score(X_encoded, y)
    print(f"max_depth={depth}: {accuracy:.2%}")




    #4 REFLECTION QUESTIONS:
    #1. This task would be supervised as it useslabeled data to train the model. It learns patterns from the data we give it
    #2. Yes, it could. If the data was useless then it would simply not appear on the tree.
    #3. Deep trees could potentially memorize trainingdata rather than learning the pattern. The shallow tree would have less to work with and could possibly make better predictions.
    #4. A bank could use a decision tree. For example, whether someone gets a loan or not. They could look at credit score, income, and debt as factors. The labels would be as simple as approved or denied.