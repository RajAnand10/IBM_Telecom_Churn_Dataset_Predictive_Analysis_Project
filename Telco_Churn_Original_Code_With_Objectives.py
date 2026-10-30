# ====================================================================
# TELCO CUSTOMER CHURN PREDICTION  (Predictive Analytics Project)
# Business question: Which telecom customers are likely to leave (churn)?
# Models used (same as taught in class):
#   1. Logistic Regression   2. KNN   3. Naive Bayes   4. SVM
# ====================================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# --------------------------------------------------------------------
# STEP 1: LOAD AND EXPLORE THE DATA  (Practical 1)
# --------------------------------------------------------------------
df = pd.read_csv(r'/Users/rajanand/Desktop/PREDICTIVE ANALYSIS/telco.csv')   # put telco.csv in the same folder, or give the full path

print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
df.info()
print(df.describe())
print(df.isna().sum())

# ================================================================
# OBJECTIVE 1: Find the percentage of customers who leave the company.
# The following code calculates how many customers churned and stayed.
# ================================================================
# Target column: Churn Label (Yes = customer left, No = customer stayed)
print(df['Churn Label'].value_counts())
print(df['Churn Label'].value_counts(normalize=True) * 100)

# Pie chart - how many customers churned? (taught in Unit 3 charts)
plt.figure(figsize=(5, 5))
df['Churn Label'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Churned vs Stayed Customers")
plt.ylabel("")   # Hide the y-label
plt.savefig("churn_pie_chart.png")
plt.show()

# Histogram - distribution of tenure (how long customers have stayed)
plt.figure(figsize=(8, 4))
plt.hist(df['Tenure in Months'], bins=20, color='purple', edgecolor='black', alpha=0.9)
plt.title('Histogram of Tenure in Months')
plt.xlabel('Tenure in Months')
plt.ylabel('Frequency')
plt.savefig("hist_tenure.png")
plt.show()

# Histogram - distribution of monthly charge
plt.figure(figsize=(8, 4))
plt.hist(df['Monthly Charge'], bins=20, color='orange', edgecolor='black', alpha=0.9)
plt.title('Histogram of Monthly Charge')
plt.xlabel('Monthly Charge')
plt.ylabel('Frequency')
plt.savefig("hist_monthly_charge.png")
plt.show()

# --------------------------------------------------------------------
# STEP 2: HANDLE MISSING VALUES  (Practical 1)
# --------------------------------------------------------------------
# Offer and Internet Type are blank when the customer has NO offer / NO internet.
# So the blank is not an error, it means "None". We fill it with the word "None".
df['Offer'] = df['Offer'].fillna('None')
df['Internet Type'] = df['Internet Type'].fillna('None')

# Churn Category and Churn Reason are blank for customers who did not churn.
# We will drop these two columns in Step 4, so no filling is needed.

# Numeric columns: fill with median (as taught). None are missing here, but it is safe.
nums_cols = ['Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges']
for col in nums_cols:
    df[col] = df[col].fillna(df[col].median())

# ================================================================
# OBJECTIVE 2: Identify which contract types have the highest churn rate.
# The following analysis compares churn rates across contract types
# and other customer groups, including internet type, offers and payment method.
# ================================================================

# --------------------------------------------------------------------
# STEP 3: EDA - FINDINGS (which groups of customers churn more?)
# --------------------------------------------------------------------
# Make a 0/1 churn column just for the charts (No = 0, Yes = 1)
le = LabelEncoder()
df['Churn_Num'] = le.fit_transform(df['Churn Label'])

# BAR CHARTS: bar height = average of Churn_Num = churn rate (0.45 means 45% left).
# The thin black line on each bar is the confidence interval (CI).

# Finding 1: Contract type
plt.figure(figsize=(8, 4))
sns.barplot(x='Contract', y='Churn_Num', data=df, estimator=np.mean, color='skyblue')
plt.title("Churn Rate by Contract")
plt.xlabel('Contract')
plt.ylabel('Churn Rate (0.45 = 45%)')
plt.savefig("churn_by_contract.png")
plt.show()

# Finding 2: Internet type
plt.figure(figsize=(8, 4))
sns.barplot(x='Internet Type', y='Churn_Num', data=df, estimator=np.mean, color='skyblue')
plt.title("Churn Rate by Internet Type")
plt.xlabel('Internet Type')
plt.ylabel('Churn Rate (0.45 = 45%)')
plt.savefig("churn_by_internet.png")
plt.show()

# Finding 3: Premium Tech Support
plt.figure(figsize=(8, 4))
sns.barplot(x='Premium Tech Support', y='Churn_Num', data=df, estimator=np.mean, color='skyblue')
plt.title("Churn Rate by Premium Tech Support")
plt.xlabel('Premium Tech Support')
plt.ylabel('Churn Rate (0.45 = 45%)')
plt.savefig("churn_by_tech_support.png")
plt.show()

# Finding 4: Offer
plt.figure(figsize=(8, 4))
sns.barplot(x='Offer', y='Churn_Num', data=df, estimator=np.mean, color='skyblue')
plt.title("Churn Rate by Offer")
plt.xlabel('Offer')
plt.ylabel('Churn Rate (0.45 = 45%)')
plt.savefig("churn_by_offer.png")
plt.show()

# Finding 5: Payment method
plt.figure(figsize=(8, 4))
sns.barplot(x='Payment Method', y='Churn_Num', data=df, estimator=np.mean, color='skyblue')
plt.title("Churn Rate by Payment Method")
plt.xlabel('Payment Method')
plt.ylabel('Churn Rate (0.45 = 45%)')
plt.savefig("churn_by_payment.png")
plt.show()

# LINE CHART: churn rate with respect to tenure (numeric vs timeline)
churn_by_tenure = df.groupby('Tenure in Months')['Churn_Num'].mean() * 100
plt.figure(figsize=(8, 4))
plt.plot(churn_by_tenure.index, churn_by_tenure.values, marker='o', linestyle='--', color='b')
plt.title('Churn Rate (%) by Tenure')
plt.xlabel('Tenure in Months')
plt.ylabel('Churn Rate (%)')
plt.grid()
plt.savefig("line_churn_tenure.png")
plt.show()

# ================================================================
# OBJECTIVE 3: Investigate whether customer tenure and monthly charges
# are related to churn.
# The following charts compare tenure and monthly charges for customers
# who left the company and customers who stayed.
# ================================================================

# BOX PLOTS: compare numeric columns for churned and stayed customers

# Finding 6: Tenure - churned customers have a much lower tenure
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Churn Label', y='Tenure in Months')
plt.title("Tenure of Churned and Stayed Customers")
plt.savefig("box_tenure.png")
plt.show()

# Finding 7: Monthly Charge - churned customers pay more per month
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Churn Label', y='Monthly Charge')
plt.title("Monthly Charge of Churned and Stayed Customers")
plt.savefig("box_monthly_charge.png")
plt.show()

# Finding 8: Monthly Charge for each contract, split by churn (box plot with hue)
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='Contract', y='Monthly Charge', hue='Churn Label')
plt.title("Monthly Charge by Contract (Stayed vs Churned)")
plt.savefig("box_charge_contract.png")
plt.show()

# SCATTER PLOT: tenure vs monthly charge, coloured by churn
plt.figure(figsize=(8, 4))
sns.scatterplot(x='Tenure in Months', y='Monthly Charge', hue='Churn Label',
                data=df, alpha=0.6, palette='coolwarm')
plt.title('Scatter Plot of Tenure vs Monthly Charge')
plt.grid()
plt.savefig("scatter_tenure_charge.png")
plt.show()

# PAIR PLOT: relationship between all main numeric columns at once, coloured by churn
sns.pairplot(df[['Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges', 'Churn Label']],
             hue='Churn Label')
plt.savefig("pairplot.png")
plt.show()

# Why we must NOT use Satisfaction Score as a feature
plt.figure(figsize=(8, 4))
sns.barplot(x='Satisfaction Score', y='Churn_Num', data=df, estimator=np.mean, color='skyblue')
plt.title("Churn Rate by Satisfaction Score")
plt.xlabel('Satisfaction Score')
plt.ylabel('Churn Rate (1.0 = 100%)')
plt.savefig("churn_by_satisfaction.png")
plt.show()

# Score 1-2 -> 100% churn, Score 4-5 -> 0% churn. It gives away the answer (data leakage).

df = df.drop(columns=['Churn_Num'])   # remove the helper column

# --------------------------------------------------------------------
# STEP 4: DROP IRRELEVANT COLUMNS  (50 columns -> 30 features + 1 target)
# --------------------------------------------------------------------
# (a) ID, constant and location columns (no predictive value)
drop_id_location = ['Customer ID', 'Country', 'State', 'Quarter',
                    'City', 'Zip Code', 'Latitude', 'Longitude']

# (b) Outcome columns - they tell us the answer (data leakage)
drop_leakage = ['Customer Status', 'Churn Score', 'Churn Category',
                'Churn Reason', 'Satisfaction Score']

# (c) Redundant columns - same information already present in another column
drop_redundant = ['Under 30', 'Senior Citizen', 'Dependents',
                  'Referred a Friend', 'Internet Service', 'Total Revenue']

df = df.drop(columns=drop_id_location + drop_leakage + drop_redundant)
print("\nShape after dropping columns:", df.shape)   # 31 columns = 30 features + 1 target

# --------------------------------------------------------------------
# STEP 5: ENCODE CATEGORICAL COLUMNS  (Practical 1)
# --------------------------------------------------------------------
# Target: Churn Label  (No = 0, Yes = 1)   (le was created in Step 3)
df['Churn Label'] = le.fit_transform(df['Churn Label'])
print(le.classes_)

# Columns with only 2 values (Yes/No, Male/Female): LabelEncoder
binary_cols = ['Gender', 'Married', 'Phone Service', 'Multiple Lines',
               'Online Security', 'Online Backup', 'Device Protection Plan',
               'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
               'Streaming Music', 'Unlimited Data', 'Paperless Billing']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Columns with 3 or more values: get_dummies (One-Hot).
# LabelEncoder would give them a fake order (e.g. 0 < 1 < 2), so we use dummies here.
multi_cols = ['Offer', 'Internet Type', 'Contract', 'Payment Method']
df = pd.get_dummies(df, columns=multi_cols, drop_first=True, dtype=int)

df.info()

# Correlation heatmap of the main numeric columns (as taught)
corr = df[['Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges',
           'Number of Referrals', 'CLTV', 'Churn Label']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# --------------------------------------------------------------------
# STEP 6: FEATURES, LABEL, TRAIN-TEST SPLIT AND SCALING
# --------------------------------------------------------------------
X = df.drop('Churn Label', axis=1)   # X = features (input)
y = df['Churn Label']                # y = label (output)
print("Number of features:", X.shape[1])

# stratify=y keeps the same churn % in train and test (important for imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)   # for training data
X_test = Scaler.transform(X_test)         # for testing data

# ================================================================
# OBJECTIVE 4: Predict which customers are likely to leave.
# The following code trains four models to predict customer churn.
# ================================================================

# --------------------------------------------------------------------
# STEP 7: BUILD THE 4 MODELS
# --------------------------------------------------------------------

# ---------- MODEL 1: LOGISTIC REGRESSION ----------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
prob_log = log_model.predict_proba(X_test)[:, 1]

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# ---------- MODEL 2: KNN ----------
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')   # Euclidean distance
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
prob_knn = knn.predict_proba(X_test)[:, 1]

print("\n===== KNN =====")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# ---------- MODEL 3: NAIVE BAYES ----------
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
prob_nb = nb_model.predict_proba(X_test)[:, 1]

print("\n===== NAIVE BAYES =====")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

# ---------- MODEL 4: SVM ----------
svm_lin = SVC(kernel='linear', random_state=42)
svm_lin.fit(X_train, y_train)
y_pred_svm = svm_lin.predict(X_test)
prob_svm = svm_lin.decision_function(X_test)   # distance from the boundary (used for AUC)

print("\n===== SVM (LINEAR) =====")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

# ================================================================
# OBJECTIVE 5: Determine which model predicts churn most effectively.
# The following code compares Accuracy, Precision, Recall, F1-score,
# and ROC-AUC for all four models.
# ================================================================

# --------------------------------------------------------------------
# STEP 8: COMPARE ALL 4 MODELS
# --------------------------------------------------------------------
results = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN", "Naive Bayes", "SVM"],
    "Accuracy": [accuracy_score(y_test, y_pred_log), accuracy_score(y_test, y_pred_knn),
                 accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_svm)],
    "Precision": [precision_score(y_test, y_pred_log), precision_score(y_test, y_pred_knn),
                  precision_score(y_test, y_pred_nb), precision_score(y_test, y_pred_svm)],
    "Recall": [recall_score(y_test, y_pred_log), recall_score(y_test, y_pred_knn),
               recall_score(y_test, y_pred_nb), recall_score(y_test, y_pred_svm)],
    "F1 Score": [f1_score(y_test, y_pred_log), f1_score(y_test, y_pred_knn),
                 f1_score(y_test, y_pred_nb), f1_score(y_test, y_pred_svm)],
    "AUC": [roc_auc_score(y_test, prob_log), roc_auc_score(y_test, prob_knn),
            roc_auc_score(y_test, prob_nb), roc_auc_score(y_test, prob_svm)]})
results = results.round(3)

print("\n===== MODEL COMPARISON =====")
print(results)
print("\nSorted by F1 Score (best model on top):")
print(results.sort_values("F1 Score", ascending=False))
results.to_csv("model_comparison.csv", index=False)

# Baseline: a model that always says "No churn"
print("\nBaseline accuracy (always predict No):", round(1 - y_test.mean(), 3))

# Bar charts (higher bar = better model)
plt.figure(figsize=(8, 4))
sns.barplot(x='Model', y='F1 Score', data=results, color='skyblue')
plt.title("F1 Score of Each Model")
plt.ylim(0, 1)
plt.savefig("compare_f1.png")
plt.show()

plt.figure(figsize=(8, 4))
sns.barplot(x='Model', y='Recall', data=results, color='skyblue')
plt.title("Recall of Each Model")
plt.ylim(0, 1)
plt.savefig("compare_recall.png")
plt.show()

plt.figure(figsize=(8, 4))
sns.barplot(x='Model', y='AUC', data=results, color='skyblue')
plt.title("AUC of Each Model")
plt.ylim(0, 1)
plt.savefig("compare_auc.png")
plt.show()

# Confusion matrices (rows = actual, columns = predicted; 0 = No churn, 1 = Churn)
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("cm_logistic.png")
plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("cm_knn.png")
plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("cm_naive_bayes.png")
plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("cm_svm.png")
plt.show()

# ROC curves (the curve closer to the top-left corner is the better model)
fpr_log, tpr_log, _ = roc_curve(y_test, prob_log)
fpr_knn, tpr_knn, _ = roc_curve(y_test, prob_knn)
fpr_nb, tpr_nb, _ = roc_curve(y_test, prob_nb)
fpr_svm, tpr_svm, _ = roc_curve(y_test, prob_svm)

plt.figure(figsize=(7, 6))
plt.plot(fpr_log, tpr_log, label="Logistic Regression")
plt.plot(fpr_knn, tpr_knn, label="KNN")
plt.plot(fpr_nb, tpr_nb, label="Naive Bayes")
plt.plot(fpr_svm, tpr_svm, label="SVM")
plt.plot([0, 1], [0, 1], 'k--', label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.savefig("roc_curves.png")
plt.show()

# --------------------------------------------------------------------
# STEP 9: WHICH FACTORS AFFECT CHURN? (Logistic Regression coefficients)
# --------------------------------------------------------------------
coef = pd.Series(log_model.coef_[0], index=X.columns)
coef = coef.sort_values()

print("\n5 factors that REDUCE churn (most negative):")
print(coef.head(5))
print("\n5 factors that INCREASE churn (most positive):")
print(coef.tail(5))

plt.figure(figsize=(8, 4))
sns.barplot(x=coef.head(5).values, y=coef.head(5).index, color='skyblue')
plt.title("5 Factors that REDUCE Churn")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.savefig("factors_reduce.png")
plt.show()

plt.figure(figsize=(8, 4))
sns.barplot(x=coef.tail(5).values, y=coef.tail(5).index, color='skyblue')
plt.title("5 Factors that INCREASE Churn")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.savefig("factors_increase.png")
plt.show()

# --------------------------------------------------------------------
# STEP 10: CONCLUSION
# --------------------------------------------------------------------
print("\n===== CONCLUSION =====")
print("Only about 26% of customers churn, so accuracy alone is misleading.")
print("The model on top of the F1 table above is the best overall model.")
print("Also check Recall (how many real churners we catch) and AUC (overall ranking quality).")
