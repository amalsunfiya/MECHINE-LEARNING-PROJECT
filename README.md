# HEART DISEASE PREDICTION USING ML

## Aim

To analyze patient health data and predict whether a person has **heart disease or is normal** using machine learning algorithms.

## Project Overview

This project focuses on building a predictive model that helps in early detection of heart disease. It involves:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Statistical analysis of features
* Training multiple Machine Learning models
* Comparing model performances using evaluation metrics
* Selecting the best-performing algorithm
* Testing the final model with real-time input data

The goal is to develop an efficient and accurate system that can assist in medical decision-making.

##  Dataset

The dataset used in this project is taken from Kaggle:
 **Heart Disease Prediction Dataset**
[https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction](https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction)

It contains various medical attributes such as age, gender, chest pain type, blood pressure, cholesterol level, fasting blood sugar, ECG results, maximum heart rate, and more.

## Project Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Model Training
6. Performance Evaluation
7. Best Model Selection
8. Real-Time Prediction Testing

## Machine Learning Algorithms Used

The following four algorithms were implemented and compared:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest

## Performance Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The best-performing algorithm was selected based on these metrics and used for final prediction.

## Key Takeaways

* Machine Learning can effectively predict heart disease with high accuracy.
* Proper data preprocessing significantly improves model performance.
* Random Forest / Logistic Regression (update based on your result) showed the best performance.
* Real-time testing confirms the efficiency of the trained model.
* The system can assist doctors and healthcare professionals in early diagnosis.

## Technologies & Tools Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Google Colab / VS Code / Jupyter Notebook

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
```

2. Navigate to the project directory

```bash
cd heart-disease-prediction
```

3. Install required libraries

```bash
pip install -r requirements.txt
```

4. Run the notebook or Python script

```bash
python main.py
```

## Results

The trained model successfully predicts whether a patient has heart disease or not based on medical parameters. The selected model achieves high accuracy and reliability.

## Future Enhancements

* Deploy the model using Flask or Streamlit
* Include more real-world hospital datasets
* Apply Deep Learning models for higher accuracy
* Develop a full-fledged web application

## License

This project is open-source and available under the MIT License.
