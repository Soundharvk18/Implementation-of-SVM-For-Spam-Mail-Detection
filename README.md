# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Collect and label email data as spam or non-spam (ham).

2.Preprocess text using tokenization, stop-word removal, and vectorization (e.g., TF-IDF).

3.Split dataset into training and testing sets.

4.Train the SVM classifier to find the optimal hyperplane separating spam and ham.

5.Test the model and use it to predict whether new emails are spam or not. 

## Program:

Program to implement the SVM For Spam Mail Detection..

Developed by: Jeya soundhar P

RegisterNumber:  212224040138

```
# Exp 11 – SVM for Spam Mail Detection with Visualization

# 1. Import Required Libraries
import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Detect File Encoding
file_path ="/content/spam.csv"    # use "/content/spam.csv" for Colab

with open(file_path, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

print("Detected Encoding:", result)

# 3. Load Dataset
data = pd.read_csv(file_path, encoding=result['encoding'])

# 4. Basic Data Exploration
print(data.head())
print(data.info())
print(data.isnull().sum())

# 5. Visualization: Spam vs Ham Distribution
plt.figure(figsize=(5,4))
sns.countplot(x=data['v1'])
plt.title("Distribution of Spam and Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()

# 6. Message Length Visualization
data['msg_length'] = data['v2'].apply(len)

plt.figure(figsize=(6,4))
sns.histplot(data=data, x='msg_length', hue='v1', bins=50, kde=True)
plt.title("Message Length Distribution (Spam vs Ham)")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

# 7. Feature and Target Selection
x = data['v2'].values     # messages
y = data['v1'].values     # labels

# 8. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# 9. Text Vectorization (Bag of Words)
cv = CountVectorizer()
x_train_vec = cv.fit_transform(x_train)
x_test_vec = cv.transform(x_test)

# 10. Initialize and Train SVM
svc = SVC(kernel='linear')
svc.fit(x_train_vec, y_train)

# 11. Prediction
y_pred = svc.predict(x_test_vec)

# 12. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 13. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix – SVM Spam Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

```


## Output:
<img width="778" height="477" alt="image" src="https://github.com/user-attachments/assets/ecab0f72-e601-4d1d-a487-1aa687db8862" />
<img width="557" height="675" alt="image" src="https://github.com/user-attachments/assets/24cb47e2-d2a4-4343-8bc6-9ada75d568c1" />
<img width="473" height="520" alt="image" src="https://github.com/user-attachments/assets/fc01e11b-6a3e-443c-9a47-e9fc479e365e" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
