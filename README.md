Chronic Disease Prediction System

🌟 Project Overview

This repository hosts the machine learning models and associated code for a real-time, AI-powered computer-aided diagnosis system designed for the early detection and risk assessment of seven major chronic diseases: Diabetes, Liver Disease, Lung Disease, Lung Cancer, Breast Cancer, and Heart Disease.

Developed as a graduation project, this system leverages advanced machine learning algorithms and innovative feature engineering techniques to provide accurate predictions, personalized risk scores, and actionable health recommendations. Our goal is to empower patients to take control of their health and assist healthcare providers in delivering faster diagnoses and personalized care, ultimately mitigating health risks and improving patient outcomes.


Key Features:

Multi-Disease Prediction: Models for Diabetes, Heart Disease, Liver Disease, Lung Disease, Lung Cancer, and Breast Cancer.

Advanced ML Algorithms: Utilizes a blend of traditional ML (XGBoost, Random Forest, SVM) and ensemble methods (Voting Classifiers) tailored for specific data types (tabular and image-based).

Intelligent Feature Engineering: Employs techniques like PCA, color histograms, and statistical feature selection to extract meaningful patterns from diverse medical datasets.

Data-Driven Insights: Provides predictive analytics and personalized risk assessments.

Comprehensive System Design: Includes a conceptual framework for user management, health monitoring, lifestyle recommendations, clinic management, emergency support, and reporting.


🧠 AI/ML Methodology


Our approach focuses on building robust and efficient AI models that can be deployed in real-world healthcare settings. We've meticulously addressed challenges such as data imbalance, feature relevance, and model interpretability.

Core AI Principles Applied:

Supervised Learning: All models are trained on labeled datasets to learn patterns for classification.

Feature Engineering & Dimensionality Reduction: Techniques like Principal Component Analysis (PCA), color histograms, and statistical feature selection (SelectKBest, ExtraTreesClassifier) are used to transform raw data into a format suitable for effective model training and to reduce computational complexity.

Ensemble Learning: For image-based tasks (Lung Disease, Lung Cancer, Breast Cancer), a VotingClassifier combining RandomForestClassifier and SVC (Support Vector Machine) was chosen. This strategy enhances robustness, minimizes overfitting, and improves generalization by leveraging the strengths of multiple algorithms.

Gradient Boosting: XGBoost was selected for tabular datasets (Diabetes, Heart Disease, Liver Disease) due to its high performance, efficiency, and ability to handle complex non-linear relationships.

Data Augmentation: Applied to image datasets to increase data diversity and improve model generalization, reducing overfitting.

Class Imbalance Handling: Techniques like NearMiss (undersampling) and SMOTE (oversampling) were employed to address skewed class distributions in datasets, ensuring models do not bias towards the majority class.



Model Performance Highlights:


Our models achieved high accuracy across all diseases, demonstrating the effectiveness of the chosen AI methodologies:

Diabetes Prediction: 92.03% Accuracy (XGBoost)

Heart Disease Prediction: 95.21% Accuracy (XGBoost)

Liver Disease Prediction: 96.00% Accuracy (XGBoost)

Breast Cancer Detection: 94.61% Accuracy (Voting Classifier)

Lung Disease Detection: 96.83% Accuracy (Voting Classifier)

Lung Cancer Detection: 97.54% Accuracy (Voting Classifier)

(For detailed performance metrics including Precision, Recall, and F1-Score for each class, please refer to the individual model cards in the models/ directory.)




📂 Repository Structure


.
├── notebooks/
│   ├── new_diabetes.ipynb
│   ├── New_Heart_disease.ipynb
│   ├── new_liver_disease.ipynb
│   ├── new_lung_disease.ipynb
│   ├── newLung_cancer.ipynb
│   └── Last_breast_cancer.ipynb
├── models/
│   ├── diabetes_prediction_model_card.md
│   ├── heart_disease_prediction_model_card.md
│   ├── liver_disease_prediction_model_card.md
│   ├── lung_disease_detection_model_card.md
│   ├── lung_cancer_detection_model_card.md
│   ├── breast_cancer_detection_model_card.md
│   ├── voting.pkl                  # Example: Breast Cancer/Lung Disease/Lung Cancer model
│   ├── scaler.pkl                  # Example: Scaler for various models
│   ├── pca.pkl                     # Example: PCA for image models
│   ├── new_xgboost_model.pkl       # Example: Diabetes model
│   ├── new_XGB_2.pkl               # Example: Heart Disease model
│   └── feature_names.pkl           # Example: Feature names for numerical models
├── README.md
├── .gitignore
└── LICENSE





🚀 Getting Started

To explore the models and notebooks:


Clone the repository:


git clone https://github.com/mohamedeltaieb/Chronic-Disease-Prediction-System.git
cd Chronic-Disease-Prediction-System

Install dependencies:
It's recommended to create a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

(You'll need to create a requirements.txt file. See "Dependencies" section below.)

Explore Notebooks:
Open the Jupyter notebooks in the notebooks/ directory to see the full data preprocessing, model training, and evaluation pipelines.

jupyter notebook

Inspect Models:
Refer to the Markdown files in the models/ directory for detailed explanations of each trained model.


🛠️ Dependencies


pandas
numpy
scikit-learn
xgboost
lightgbm
imbalanced-learn
opencv-python
scikit-image
matplotlib
seaborn
kagglehub



🤝 Contribution

This project was developed as a graduation project. For any inquiries or potential collaborations, please reach out to the project team members.


📄 License

This project is licensed under the MIT License. See the LICENSE file for details.


Project Team:




Habiba Ahmed Hassan (AI Engineer) [Habiba480][https://github.com/Habiba480/Habiba480]


Mohamed Mostafa Ahmed Eltaieb (AI Engineer)


Mohamed Elsayed Youssef (Backend)


Mohamed Khalid Mohamed (Devops and Backend)


Tasneem Mostafa Mohamed (Frontend)


Zienab Youssef Abdel Kareem (Frontend)





Under Supervision of:


Dr. Ibrahim Shawky


Dr. Mohamed Abdel Hamid


Dr. Osama Abu Elnasr
