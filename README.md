
Heart Disease Prediction Using Machine Learning (Logistic Regression) 🫀💻
📌 Project Overview
This project predicts the risk of heart disease in patients based on various health parameters using Logistic Regression. The model is trained on the Framingham Heart Study dataset, which contains features such as age, cholesterol levels, blood pressure, smoking status, and diabetes history.

🗂️ Dataset
Dataset Name: Framingham Heart Study
Source: Kaggle
Target Variable: TenYearCHD (0 = No Heart Disease, 1 = Risk of Heart Disease)
🔧 Technologies Used
Google Colab (for cloud-based model training)
Python Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
📊 Machine Learning Model Used
Logistic Regression (Binary Classification Model)
🛠️ Installation & Usage
1️⃣ Open in Google Colab
Click the button below to open the notebook:

2️⃣ Load the Dataset in Google Colab
from google.colab import files
uploaded = files.upload()
3️⃣ Install Dependencies
!pip install pandas numpy scikit-learn matplotlib seaborn
4️⃣ Run the Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("framingham.csv")

# Data Preprocessing
df.fillna(df.mean(), inplace=True)

# Splitting into features and target
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions & Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
📈 Model Performance
Logistic Regression Accuracy: ~85%
Model trained on Google Colab using scikit-learn.
📌 Future Improvements
Tune hyperparameters for better performance.
Experiment with other models (Random Forest, SVM, Deep Learning).
Deploy the model using Flask/Streamlit for real-world use.
📩 Contact
GitHub: Vishalchau2003
