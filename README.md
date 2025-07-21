# 💼 Employee Salary Prediction using Machine Learning

This project uses Machine Learning to predict whether an individual earns more than \$50K or less than or equal to \$50K annually, based on various demographic features.

---

## 📌 Problem Statement

Predicting employee salaries manually can be inaccurate and biased. Our goal is to build a machine learning model that predicts salary class (`<=50K` or `>50K`) using features like education, occupation, workclass, etc. This can help HR teams make faster and fairer decisions.

---

## 📊 Dataset

- **Source:** [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **File:** `adult.csv`
- **Attributes Include:**
  - Age
  - Workclass
  - Education
  - Occupation
  - Marital Status
  - Hours per week
  - Native country
  - Salary (Target variable)

---

## ⚙️ Technologies Used

- **Language:** Python  
- **Libraries:**
  - `pandas`, `numpy` – data processing  
  - `seaborn`, `matplotlib` – data visualization  
  - `sklearn` – machine learning models  

---

## 🧠 Models Implemented

- Logistic Regression ✅ *(Best Performing)*
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- Naive Bayes Classifier  

---

## 🔄 Workflow

1. Load and explore the dataset  
2. Clean and preprocess the data  
   - Handle missing values  
   - Encode categorical variables  
3. Split into training and test sets  
4. Train multiple ML models  
5. Evaluate model performance using accuracy, confusion matrix, and classification report  
6. Choose the best model (Logistic Regression)

---

## 📈 Results

- **Best Accuracy:** ~85% with Logistic Regression  
- **Evaluation Metrics Used:**
  - Confusion Matrix
  - Accuracy Score
  - Precision, Recall, F1-score

---

## 🚀 Deployment

This model can be easily deployed using:
- Flask / FastAPI for creating web APIs
- Streamlit / Gradio for a simple UI

---



## 🔗 GitHub Link

[👉 Click here to view the full project on GitHub](https://github.com/KSSRUTHI/Employee-Salary-Prediction_IBM-SkillsBuild/tree/main)  
*(Update this with your actual repo link)*

---

## 📝 Conclusion

- Successfully predicted salary brackets using machine learning  
- Logistic Regression offered a good balance between performance and interpretability  
- The project helps automate salary classification and supports HR decision-making
---

## 📚 References

- UCI Machine Learning Repository  
- Scikit-learn Documentation  
- TowardsDataScience articles on Salary Prediction  
- Python Official Documentation

---

## 👩‍💻 Author

**Sruthi Sai Prabha K S**  

