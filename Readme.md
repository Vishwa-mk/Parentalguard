📌 **Introduction**
ParentalGuard is an intelligent, web-based maternal health surveillance system designed to predict and monitor pregnancy-related health risks using machine learning. Maternal health complications such as preeclampsia and gestational diabetes remain leading causes of maternal and neonatal mortality, especially in rural and underserved regions.

This project leverages supervised machine learning, real-time clinical data input, and explainable AI techniques to classify maternal risk levels into Low, Medium, or High categories. By enabling early detection and continuous monitoring, ParentalGuard supports timely medical intervention and contributes to safer pregnancies and improved healthcare outcomes.


🧠 **Methodology**
The ParentalGuard system follows a structured and data-driven workflow:

*Data Collection*
Uses historical maternal health datasets containing physiological parameters such as:
-Age
-Systolic Blood Pressure
-Diastolic Blood Pressure
-Blood Sugar Level
-Body Temperature
-Heart Rate

*Data Preprocessing*
-Handling missing values
-Feature selection and normalization
-Encoding target labels (Low, Medium, High risk)

*Model Training*
-Supervised learning using XGBoost and Random Forest
-Model selection based on accuracy and performance
-Achieved accuracy of approximately 93%

*Risk Prediction*
-New patient data is passed to the trained model
-The model predicts the maternal risk category
-Explainable AI (XAI)
-SHAP (SHapley Additive Explanations) is used to interpret predictions
-Helps clinicians understand which factors influenced the risk level

*Data Storage & Visualization*
-Predictions and patient data are securely stored in MySQL
-Graphical insights using bar charts and line graphs for trend analysis


🛠️ **Tech Stack Used**
🔹 Programming & ML
Python3
Pandas, NumPy
Scikit-learn
XGBoost
SHAP (Explainable AI)

🔹 Backend
Flask
Pickle (Model serialization)

🔹 Frontend
HTML
CSS
JavaScript

🔹 Database
MySQL

🔹 Visualization
Matplotlib

🔹 Tools & Environment
VS Code
Jupyter Notebook
Google Chrome
Windows OS


▶️ **How to Run the Program**

Step 1: Install Required Libraries
pip install -r requirements.txt

Step 2: Set Up the Database
Install MySQL
Create a database (e.g., parentalguard_db)
Import the provided SQL schema

Step 3: Run the Application
python app.py

Step 4: Access the Web App
Open your browser and go to:
http://127.0.0.1:5000/


✅ **Advantages of the Project**
-Early detection of maternal health risks
-High prediction accuracy using advanced ML algorithms
-Explainable AI improves trust and clinical acceptance
-Secure data storage with patient history tracking
-User-friendly and responsive web interface
-Supports decision-making for healthcare professionals
-Scalable and future-ready for IoT and mobile integration


🌍 **Applications of the Project**
-Hospitals and maternity clinics
-Rural and remote healthcare centers
-Government maternal health monitoring programs
-Community health worker support systems
-Academic and medical research
-Integration with IoT-based health monitoring devices


🎯 **Conclusion**

ParentalGuard demonstrates how machine learning and predictive analytics can transform maternal healthcare by enabling early risk identification, continuous monitoring, and data-driven decision-making. The system is scalable, explainable, and impactful, making it a strong step toward reducing preventable maternal and neonatal complications.