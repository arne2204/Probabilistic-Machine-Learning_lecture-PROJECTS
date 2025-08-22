import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from pandas.plotting import scatter_matrix
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# load dataset
df = pd.read_csv('../Data/ufc-master.csv')

# first data exploration
print("Data Insights:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

print("\nSummary Statistics:")
print(df.describe())

# Create the 'Results' directory if it doesn't exist
if not os.path.exists('../Results'):
    os.makedirs('../Results')

plt.figure(figsize=(24, 12))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Data Heatmap", fontsize=16)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
# # plt.show()
plt.savefig(os.path.join('../Results', 'missing_data_heatmap.png'))

# select and visualize physical attributes
physical_features = [
    'RedHeightCms', 'BlueHeightCms',
    'RedReachCms', 'BlueReachCms',
    'RedWeightLbs', 'BlueWeightLbs'
]

# Histograms
fig = df[physical_features].hist(bins=20, figsize=(18, 10))
fig[0][0].figure.suptitle("Distributions of Physical Attributes", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join('../Results', 'physical_attributes_histograms.png'))
plt.close()



# correlation heatmap
plt.figure(figsize=(24, 12))
sns.heatmap(df[physical_features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Physical Attributes")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'physical_attributes_correlation.png'))

# create scatter-Matrix
scatter_matrix(df[physical_features],
               figsize=(12, 12),
               diagonal='hist',
               alpha=0.8,
               marker='o',
               grid=True,
               color='teal')

plt.suptitle("Scatter Matrix of Physical Attributes")
# plt.show()
plt.savefig(os.path.join('../Results', 'physical_attributes_scatter_matrix.png'))
plt.close()
# create different features
df_filtered = df.dropna(subset=physical_features + ['Winner'])
df_filtered['height_diff'] = df_filtered['RedHeightCms'] - df_filtered['BlueHeightCms']
df_filtered['reach_diff'] = df_filtered['RedReachCms'] - df_filtered['BlueReachCms']
df_filtered['weight_diff'] = df_filtered['RedWeightLbs'] - df_filtered['BlueWeightLbs']

# target variable: 1 if red wins, 0 if blue
X = df_filtered[['height_diff', 'reach_diff', 'weight_diff']]
y = df_filtered['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train naive bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# evaluate model

y_pred = model.predict(X_test)

report_nb = classification_report(y_test, y_pred, output_dict=True)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# confusion matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))  # Wichtig: Neue Figure erzeugen
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for naive bayes model')

plt.tight_layout()  # Layout automatisch optimieren
plt.savefig(os.path.join('../Results', 'naive_bayes_confusion_matrix.png'))
plt.close()

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

lda.fit(X_train, y_train)
qda.fit(X_train, y_train)

y_pred_lda = lda.predict(X_test)
y_pred_qda = qda.predict(X_test)

report_lda = classification_report(y_test, y_pred_lda, output_dict=True)
report_qda = classification_report(y_test, y_pred_qda, output_dict=True)

print("LDA Accuracy:", accuracy_score(y_test, y_pred_lda))
print("QDA Accuracy:", accuracy_score(y_test, y_pred_qda))

print("LDA Report:\n", classification_report(y_test, y_pred_lda))
print("QDA Report:\n", classification_report(y_test, y_pred_qda))

# Confusion Matrices
cm_lda = confusion_matrix(y_test, y_pred_lda)
cm_qda = confusion_matrix(y_test, y_pred_qda)

# Visualization with Seaborn
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_lda, annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title("Confusion Matrix - LDA")
axs[0].set_xlabel("Predicted class")
axs[0].set_ylabel("Actual class")

sns.heatmap(cm_qda, annot=True, fmt="d", cmap="Greens", ax=axs[1])
axs[1].set_title("Confusion Matrix - QDA")
axs[1].set_xlabel("Predicted class")
axs[1].set_ylabel("Actual class")

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'lda_qda_confusion_matrices.png'))
plt.close()
# GDA adjustment --> Correct class weight - Option 1: with same Priors (e. g. 50/50)

lda_equal = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
qda_equal = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5])

lda_equal.fit(X_train, y_train)
qda_equal.fit(X_train, y_train)

# Predictions & Evaluation
y_pred_lda_eq = lda_equal.predict(X_test)
y_pred_qda_eq = qda_equal.predict(X_test)

report_lda_eq = classification_report(y_test, y_pred_lda_eq, output_dict=True)
report_qda_eq = classification_report(y_test, y_pred_qda_eq, output_dict=True)

print("LDA (Equal Priors) Accuracy:", accuracy_score(y_test, y_pred_lda_eq))
print("QDA (Equal Priors) Accuracy:", accuracy_score(y_test, y_pred_qda_eq))

print("LDA Report:\n", classification_report(y_test, y_pred_lda_eq))
print("QDA Report:\n", classification_report(y_test, y_pred_qda_eq))

# Confusion Matrices
cm_lda = confusion_matrix(y_test, y_pred_lda)
cm_qda = confusion_matrix(y_test, y_pred_qda)

# Visualization with Seaborn
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_lda, annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title("Confusion Matrix - LDA: Option 1: with same Priors")
axs[0].set_xlabel("Predicted class")
axs[0].set_ylabel("Actual class")

sns.heatmap(cm_qda, annot=True, fmt="d", cmap="Greens", ax=axs[1])
axs[1].set_title("Confusion Matrix - QDA: Option 1: with same Priors")
axs[1].set_xlabel("Predicted class")
axs[1].set_ylabel("Actual class")

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'lda_qda_equal_priors_confusion_matrices.png'))
plt.close()
# GDA adjustment --> Correct class weight - Option 2: Manually adjust the decision threshold

#LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Probability for class 1 (e.g. Blue)
probs_lda = lda.predict_proba(X_test)
probs_qda = qda.predict_proba(X_test)
# Set threshold to 0.6 (instead of 0.5)
y_pred_lda_thresh = (probs_lda[:,1] > 0.6).astype(int)
y_pred_qda_thresh = (probs_qda[:, 1] > 0.6).astype(int)

report_lda_thresh = classification_report(y_test, y_pred_lda_thresh, output_dict=True)
report_qda_thresh = classification_report(y_test, y_pred_qda_thresh, output_dict=True)

# Evaluation
print("LDA (Threshold 0.6) Accuracy:", accuracy_score(y_test, y_pred_lda_thresh))
print("LDA (Threshold 0.6) Report:\n", classification_report(y_test, y_pred_lda_thresh))

print("QDA (Threshold 0.6) Accuracy:", accuracy_score(y_test, y_pred_qda_thresh))
print("QDA (Threshold 0.6) Report:\n", classification_report(y_test, y_pred_qda_thresh))

# Confusion Matrices
cm_lda = confusion_matrix(y_test, y_pred_lda_thresh)
cm_qda = confusion_matrix(y_test, y_pred_qda_thresh)

# Visualization with Seaborn
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_lda, annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title("Confusion Matrix - LDA: Option 2: Manual Threshold Adjustment")
axs[0].set_xlabel("Predicted class")
axs[0].set_ylabel("Actual class")

sns.heatmap(cm_qda, annot=True, fmt="d", cmap="Greens", ax=axs[1])
axs[1].set_title("Confusion Matrix - QDA: Option 2: Manual Threshold Adjustment")
axs[1].set_xlabel("Predicted class")
axs[1].set_ylabel("Actual class")

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'lda_qda_manual_threshold_confusion_matrices.png'))
plt.close()
# Select the first two features for visualization
X_train_2d = X_train[['height_diff', 'reach_diff']]
X_test_2d = X_test[['height_diff', 'reach_diff']]

# Train LDA model
lda_2d = LinearDiscriminantAnalysis()
lda_2d.fit(X_train_2d, y_train)

# Train LDA model with equal priors
lda_equal_2d = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
lda_equal_2d.fit(X_train_2d, y_train)

# Train LDA model for manual threshold adjustment (using default priors for training)
lda_thresh_2d = LinearDiscriminantAnalysis()
lda_thresh_2d.fit(X_train_2d, y_train)

# Create a meshgrid to plot the decision boundary
x_min, x_max = X_test_2d.iloc[:, 0].min() - 1, X_test_2d.iloc[:, 0].max() + 1
y_min, y_max = X_test_2d.iloc[:, 1].min() - 1, X_test_2d.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Define colormaps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ['#FF0000', '#00FF00']

# Plot decision boundaries for all three LDA models
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Standard LDA
Z_lda = lda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lda = Z_lda.reshape(xx.shape)
axs[0].contourf(xx, yy, Z_lda, cmap=cmap_light, alpha=0.8)
sns.scatterplot(x=X_test_2d.iloc[:, 0], y=X_test_2d.iloc[:, 1], hue=y_test,
                palette=cmap_bold, alpha=0.6, edgecolor="k", s=50, ax=axs[0])
axs[0].set_title("Standard LDA Decision Boundary")
axs[0].set_xlabel("Height Difference (Red - Blue)")
axs[0].set_ylabel("Reach Difference (Red - Blue)")

# LDA with Equal Priors
Z_lda_equal = lda_equal_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lda_equal = Z_lda_equal.reshape(xx.shape)
axs[1].contourf(xx, yy, Z_lda_equal, cmap=cmap_light, alpha=0.8)
sns.scatterplot(x=X_test_2d.iloc[:, 0], y=X_test_2d.iloc[:, 1], hue=y_test,
                palette=cmap_bold, alpha=0.6, edgecolor="k", s=50, ax=axs[1])
axs[1].set_title("LDA with Equal Priors Decision Boundary")
axs[1].set_xlabel("Height Difference (Red - Blue)")
axs[1].set_ylabel("Reach Difference (Red - Blue)")

# LDA with Manual Threshold (Threshold 0.6 for class 1)
probs_lda_thresh = lda_thresh_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z_lda_thresh = (probs_lda_thresh[:, 1] > 0.6).astype(int)
Z_lda_thresh = Z_lda_thresh.reshape(xx.shape)
axs[2].contourf(xx, yy, Z_lda_thresh, cmap=cmap_light, alpha=0.8)
sns.scatterplot(x=X_test_2d.iloc[:, 0], y=X_test_2d.iloc[:, 1], hue=y_test,
                palette=cmap_bold, alpha=0.6, edgecolor="k", s=50, ax=axs[2])
axs[2].set_title("LDA (Threshold 0.6) Decision Boundary")
axs[2].set_xlabel("Height Difference (Red - Blue)")
axs[2].set_ylabel("Reach Difference (Red - Blue)")

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'lda_decision_boundaries.png'))
plt.close()
# Select the first two features for visualization
X_train_2d = X_train[['height_diff', 'reach_diff']]
X_test_2d = X_test[['height_diff', 'reach_diff']]

# Train QDA model using only the first two features
qda_2d = QuadraticDiscriminantAnalysis()
qda_2d.fit(X_train_2d, y_train)

# Create a meshgrid to plot the decision boundary
x_min, x_max = X_test_2d.iloc[:, 0].min() - 1, X_test_2d.iloc[:, 0].max() + 1
y_min, y_max = X_test_2d.iloc[:, 1].min() - 1, X_test_2d.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict the class for each point in the meshgrid
Z = qda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create colormaps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ['#FF0000', '#00FF00'] # Use a list of colors for the scatterplot palette

# Plot the decision boundary
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

# Plot the test data points
sns.scatterplot(x=X_test_2d.iloc[:, 0], y=X_test_2d.iloc[:, 1], hue=y_test,
                palette=cmap_bold, alpha=0.6, edgecolor="k", s=50)
plt.title("QDA Decision Boundary (Height Difference vs Reach Difference)")
plt.xlabel("Height Difference (Red - Blue)")
plt.ylabel("Reach Difference (Red - Blue)")
# plt.show()
plt.savefig(os.path.join('../Results', 'qda_decision_boundary.png'))
plt.close()
# Select the first two features for visualization
X_train_2d = X_train[['height_diff', 'reach_diff']]
X_test_2d = X_test[['height_diff', 'reach_diff']]

# Train QDA model with equal priors
qda_equal_2d = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5])
qda_equal_2d.fit(X_train_2d, y_train)

# Train QDA model for manual threshold adjustment (using default priors for training)
qda_thresh_2d = QuadraticDiscriminantAnalysis()
qda_thresh_2d.fit(X_train_2d, y_train)

# Create a meshgrid to plot the decision boundary
x_min, x_max = X_test_2d.iloc[:, 0].min() - 1, X_test_2d.iloc[:, 0].max() + 1
y_min, y_max = X_test_2d.iloc[:, 1].min() - 1, X_test_2d.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Define colormaps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ['#FF0000', '#00FF00']

# Plot decision boundaries for all three QDA models
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Standard QDA
Z_qda = qda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z_qda = Z_qda.reshape(xx.shape)
axs[0].contourf(xx, yy, Z_qda, cmap=cmap_light, alpha=0.8)
sns.scatterplot(x=X_test_2d.iloc[:, 0], y=X_test_2d.iloc[:, 1], hue=y_test,
                palette=cmap_bold, alpha=0.6, edgecolor="k", s=50, ax=axs[0])
axs[0].set_title("Standard QDA Decision Boundary")
axs[0].set_xlabel("Height Difference (Red - Blue)")
axs[0].set_ylabel("Reach Difference (Red - Blue)")

# QDA with Equal Priors
Z_qda_equal = qda_equal_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z_qda_equal = Z_qda_equal.reshape(xx.shape)
axs[1].contourf(xx, yy, Z_qda_equal, cmap=cmap_light, alpha=0.8)
sns.scatterplot(x=X_test_2d.iloc[:, 0], y=X_test_2d.iloc[:, 1], hue=y_test,
                palette=cmap_bold, alpha=0.6, edgecolor="k", s=50, ax=axs[1])
axs[1].set_title("QDA with Equal Priors Decision Boundary")
axs[1].set_xlabel("Height Difference (Red - Blue)")
axs[1].set_ylabel("Reach Difference (Red - Blue)")

# QDA with Manual Threshold (Threshold 0.6 for class 1)
probs_qda_thresh = qda_thresh_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z_qda_thresh = (probs_qda_thresh[:, 1] > 0.6).astype(int)
Z_qda_thresh = Z_qda_thresh.reshape(xx.shape)
axs[2].contourf(xx, yy, Z_qda_thresh, cmap=cmap_light, alpha=0.8)
sns.scatterplot(x=X_test_2d.iloc[:, 0], y=X_test_2d.iloc[:, 1], hue=y_test,
                palette=cmap_bold, alpha=0.6, edgecolor="k", s=50, ax=axs[2])
axs[2].set_title("QDA (Threshold 0.6) Decision Boundary")
axs[2].set_xlabel("Height Difference (Red - Blue)")
axs[2].set_ylabel("Reach Difference (Red - Blue)")

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'qda_decision_boundaries.png'))
plt.close()


# Standard
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)

# Wighted
logreg_weighted = LogisticRegression(class_weight='balanced')
logreg_weighted.fit(X_train, y_train)
y_pred_log_weighted = logreg_weighted.predict(X_test)

report_log = classification_report(y_test, y_pred_log, output_dict=True)
report_log_weighted = classification_report(y_test, y_pred_log_weighted, output_dict=True)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Logistic Regression Report:\n", classification_report(y_test, y_pred_log))
print("Weighted Logistic Accuracy:", accuracy_score(y_test, y_pred_log_weighted))
print("Weighted Logistic Regression Report:\n", classification_report(y_test, y_pred_log_weighted))

# Confusion Matrices
cm_log = confusion_matrix(y_test, y_pred_log)
cm_log_weighted = confusion_matrix(y_test, y_pred_log_weighted)

# Visualization with Seaborn
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title("Confusion Matrix - Logistic Regression")
axs[0].set_xlabel("Predicted Class")
axs[0].set_ylabel("True Class")

sns.heatmap(cm_log_weighted, annot=True, fmt="d", cmap="Greens", ax=axs[1])
axs[1].set_title("Confusion Matrix - Weighted Logistic Regression")
axs[1].set_xlabel("Predicted Class")
axs[1].set_ylabel("True Class")

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'logistic_regression_confusion_matrices.png'))
plt.close()


# Random Forest initialisation
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Prdictions
y_pred_rf = rf.predict(X_test)

feature_importance = rf.feature_importances_
for name, importance in zip(X.columns, feature_importance):
    print(f"Importance of {name}: {importance:.3f}")

report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

# Accuracy & Classification Report
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest - Confusion Matrix")
# plt.show()
plt.tight_layout()
plt.savefig(os.path.join('../Results', 'random_forest_confusion_matrix.png'))
plt.close()


# Extract key figures from saved reports
modelle = [
    ("Naive Bayes", report_nb),
    ("LDA (Standard)", report_lda),
    ("QDA (Standard)", report_qda),
    ("LDA (Equal Priors)", report_lda_eq),
    ("QDA (Equal Priors)", report_qda_eq),
    ("LDA (Threshold 0.6)", report_lda_thresh),
    ("QDA (Threshold 0.6)", report_qda_thresh),
    ("LogReg (Standard)", report_log),
    ("LogReg (Weighted)", report_log_weighted),
    ("Random Forest", report_rf)
]

# Results
ergebnisse = []

for name, rpt in modelle:
    accuracy = round(rpt["accuracy"], 3)
    # Handle cases where a class might not have been predicted, resulting in missing precision/recall/f1
    precision = round(rpt["1"].get("precision", 0), 3) if "1" in rpt else None
    recall_class_1 = round(rpt["1"].get("recall", 0), 3) if "1" in rpt else None
    recall_class_0 = round(rpt["0"].get("recall", 0), 3) if "1" in rpt else None
    f1 = round(rpt["1"].get("f1-score", 0), 3) if "1" in rpt else None

    ergebnisse.append([name, accuracy, precision, recall_class_1, recall_class_0, f1])

# Create table
compare_df = pd.DataFrame(ergebnisse, columns=[
    "Modell", "Accuracy", "Precision (cl.1)", "Recall (cl.1)", "Recall (cl.0)", "F1-Score (cl.1)"
])

compare_df = compare_df.sort_values(by="F1-Score (cl.1)", ascending=False)

# Table and graph
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

table = ax.table(cellText=compare_df.values, colLabels=compare_df.columns, loc='center')

# Center alignment of cells
for (row, col), cell in table.get_celld().items():
    cell.set_text_props(ha='center', va='center')

# Font size and scaling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.tight_layout()
plt.savefig(os.path.join('../Results', 'model_compare.png'))
plt.close()
# plt.show()

# Sort by F1 score for class 1
compare_sorted = compare_df.sort_values(by="F1-Score (cl.1)", ascending=True)

# bar graph
plt.figure(figsize=(10, 6))
sns.barplot(data=compare_sorted, x="F1-Score (cl.1)", y="Modell", palette="viridis")
plt.title("F1-Score Comparison of models (Class 1)", fontsize=14)
plt.xlabel("F1-Score")
plt.ylabel("Model")
plt.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'f1_score_comparison.png'))
plt.close()
plt.figure(figsize=(8, 6))
sns.scatterplot(data=compare_df, x="Accuracy", y="F1-Score (cl.1)", hue="Modell", s=120, palette="tab10")
plt.title("Accuracy vs F1-Score (Class 1)", fontsize=14)
plt.xlabel("Accuracy")
plt.ylabel("F1-Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'accuracy_vs_f1_score.png'))
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=compare_df, x="Accuracy", y="Precision (cl.1)", hue="Modell", s=120, palette="tab10")
plt.title("Accuracy vs Precision (cl.1)", fontsize=14)
plt.xlabel("Accuracy")
plt.ylabel("Precision (cl.1)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'accuracy_vs_precision.png'))
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=compare_df, x="Accuracy", y="Recall (cl.1)", hue="Modell", s=120, palette="tab10")
plt.title("Accuracy vs Recall (cl.1)", fontsize=14)
plt.xlabel("Accuracy")
plt.ylabel("Recall (cl.1)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'accuracy_vs_recall.png'))
plt.close()
# Calculate baseline accuracy (predicting the majority class)
majority_class = y_train.mode()[0]
baseline_accuracy = (y_test == majority_class).mean()

print(f"Baseline Accuracy (predicting majority class '{majority_class}'): {baseline_accuracy:.3f}")

# Add baseline to the comparison DataFrame for Accuracy
baseline_result_acc = pd.DataFrame([["Baseline", baseline_accuracy, None, None, None, None]],
                               columns=["Modell", "Accuracy", "Precision (cl.1)", "Recall (cl.1)", "Recall (cl.0)", "F1-Score (cl.1)"])

# Use the existing compare_df and add the baseline
compare_df_with_baseline_acc = pd.concat([compare_df, baseline_result_acc], ignore_index=True)

# Sort by Accuracy for better visualization
compare_df_with_baseline_acc_sorted = compare_df_with_baseline_acc.sort_values(by="Accuracy", ascending=False)

# Create a list of colors, with 'red' for the baseline
n_models_acc = len(compare_df_with_baseline_acc_sorted)
colors_acc = sns.color_palette("viridis", n_models_acc) # Use a colormap for other bars

# Find the position of the 'Baseline' bar in the sorted data for plotting
baseline_pos_in_plot_acc = list(compare_df_with_baseline_acc_sorted['Modell']).index('Baseline')

# Change the color of the baseline bar to red
colors_acc[baseline_pos_in_plot_acc] = 'red'

# Plot the accuracy comparison
plt.figure(figsize=(12, 7))
ax_acc = sns.barplot(data=compare_df_with_baseline_acc_sorted, x="Accuracy", y="Modell", palette=colors_acc)

plt.title("Accuracy Comparison of Models vs Baseline", fontsize=16)
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'accuracy_comparison_with_baseline.png'))
plt.close()
# Extract Recall for Class 0 and Class 1 for each model from the reports
model_recall_data = []

for name, rpt in modelle:
    recall_class_0 = round(rpt["0"].get("recall", 0), 3) if "0" in rpt else None
    recall_class_1 = round(rpt["1"].get("recall", 0), 3) if "1" in rpt else None
    model_recall_data.append([name, recall_class_0, recall_class_1])

# Create a DataFrame for recall comparison
recall_df = pd.DataFrame(model_recall_data, columns=["Modell", "Recall (cl.0)", "Recall (cl.1)"])

# Melt the DataFrame for easier plotting with seaborn
recall_melted_df = recall_df.melt(id_vars="Modell", var_name="Class", value_name="Recall")

# Plot the recall comparison
plt.figure(figsize=(14, 8))
sns.barplot(data=recall_melted_df, x="Recall", y="Modell", hue="Class", palette="viridis")
plt.title("Recall Comparison by Class for Each Model", fontsize=16)
plt.xlabel("Recall")
plt.ylabel("Model")
plt.legend(title="Class")
plt.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../Results', 'recall_comparison_by_class.png'))
plt.close()