import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load the Iris dataset (Multi-class: 3 distinct flower classes)
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target  # Has classes 0, 1, and 2

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Print standard Overall Performance
print("=== Overall Performance ===")
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}\n")

# 6. Detailed Multi-Class Performance Report
# This breakdown gives precision, recall, and f1-score per individual class
print("=== Multi-Class Performance Report ===")
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(class_report)

# 7. Compute Averaged Metrics for Multi-Class Summary
# 'macro' calculates metrics for each class, and finds their unweighted mean.
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

print("=== Macro-Averaged Summary ===")
print(f"Macro Precision : {precision_macro:.2f}")
print(f"Macro Recall    : {recall_macro:.2f}")
print(f"Macro F1-Score  : {f1_macro:.2f}\n")

# 8. Generate and Plot the Multi-Class Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("=== Raw Confusion Matrix Array ===")
print(conf_matrix)

# Display visual matrix using pure Matplotlib
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues) # Uses a standard clean blue colormap

plt.title("Multi-Class Confusion Matrix (Iris Dataset)")
plt.xlabel("Predicted Flower Type")
plt.ylabel("True Flower Type")
plt.show()
