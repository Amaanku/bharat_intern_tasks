import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


titanic_data = pd.read_csv('titanic_dataset.csv')

# Preprocess the data
titanic_data.dropna(inplace=True) 
features = ['Pclass', 'Age', 'Sex'] 
X = titanic_data[features]
y = titanic_data['Survived']
 
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
print(report)
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
sns.barplot(x="Pclass", y="Survived", data=titanic_data)

plt.subplot(2, 2, 2)
sns.histplot(x="Age", hue="Survived", data=titanic_data, multiple="stack", bins=20)

plt.subplot(2, 2, 3)
sns.barplot(x="Sex", y="Survived", data=titanic_data)
plt.tight_layout()
plt.show()
