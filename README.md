
# 📘 Machine Learning - M.Sc.  
**Shahid Beheshti University**  
**Fall 2024**  
Instructor: **Dr. Farahani**  

---
## 📂 Projects Overview  

This repository contains multiple projects and assignments for the **Machine Learning** course (M.Sc., Shahid Beheshti University, Fall 2024, Instructor: Dr. Farahani).  
The projects cover various topics in supervised, unsupervised, and reinforcement learning, as well as data analysis and evaluation techniques.  

**Note on datasets:**  
All datasets used in these projects are from **Kaggle**. Some datasets are large in size, so they are **not uploaded** directly to this repository. The download links are provided inside each project's folder or README.  

---

### **Project 01 — News Popularity Prediction & Avocado Price Prediction**  
- **Files:** `project01.ipynb`, `project01.pdf`, `theory01.pdf`, `OnlineNewsPopularity.csv`, `avocado01.csv`  
- **Description:**  
  - **Part 1:** Used the **Online News Popularity** dataset to predict the number of shares for online news articles.  
    - Performed Exploratory Data Analysis (EDA) and hypothesis testing.  
    - Implemented Linear Regression, Ridge, and Lasso.  
    - Tested scaling methods and polynomial features.  
    - Used `GridSearchCV` and `RandomizedSearchCV` for hyperparameter tuning.  
  - **Part 2:** Implemented **Linear Regression with Mean Absolute Error** from scratch and compared results with Scikit-Learn using the **Avocado Price** dataset.  

---

### **Project 02 — Vehicle Insurance Claim Fraud Detection**  
- **Files:** `project02.ipynb`, `project02.pdf`, `theory02.pdf`, `fraud_oracle.csv`  
- **Description:**  
  - Built multiple classification models to detect fraudulent insurance claims.  
  - Models used: Logistic Regression, SVM, KNN, Naive Bayes, Decision Trees, Gradient Boosting, Voting Classifiers.  
  - Applied stratified cross-validation for performance evaluation.  
  - Checked dataset imbalance and applied over-sampling, under-sampling, and class-weight methods (while avoiding data leakage).  

---

### **Project 03 — ECG Heartbeat Clustering Comparison**  
- **Files:** `project03.ipynb`, `project03.pdf`, `theory03.pdf`  
- **Description:**  
  - Used the **ECG Heartbeat Categorization Dataset** for clustering tasks.  
  - Implemented K-Means, Hierarchical Clustering, and DBSCAN.  
  - Used PCA for dimensionality reduction and re-ran clustering for comparison.  
  - Evaluated using silhouette score, Davies-Bouldin index, and WCSS.  
  - Compared clustering results with ground truth labels to analyze algorithm behavior.  

---

### **Project 04 — Snake Game Q-Learning Agent**  
- **Files:** `snake-Qlearning.ipynb`, `project04.pdf`, `SnakeEnvironment.py`  
- **Description:**  
  - Designed an **intelligent agent** to play the Snake Game using **Q-Learning**.  
  - State space: snake position, food position, movement direction.  
  - Reward system: +10 for eating food, -10 for collisions, -1 for unnecessary moves.  
  - Used epsilon-greedy strategy for exploration/exploitation.  
  - Trained and evaluated the agent over multiple episodes, plotting average reward over time.  

---

### **Project 05 — Movie Recommender Systems**  
- **Files:** `project05.ipynb`, `project05.pdf`  
- **Description:**  
  - Developed a **Movie Recommendation System** using collaborative filtering techniques.  
  - Compared **user-based** and **item-based** recommendation approaches.  
  - Explored similarity measures such as cosine similarity and Pearson correlation.  
  - Evaluated recommendation quality using precision, recall, and RMSE.  

---
##  Dataset Links (from Kaggle)

- **Online News Popularity**: [Kaggle – Online News Popularity Dataset]  
- **Avocado Prices**: [Kaggle – Avocado Prices Dataset]  
- **Vehicle Insurance Fraud Detection**: [Kaggle – Vehicle Insurance Claim Fraud Detection Dataset]  
- **ECG Heartbeat Categorization**: [Kaggle – ECG Heartbeat Categorization Dataset]  
- **Movie Recommendation Data**: [Kaggle – Movie Recommendation Dataset]


  ⚠️ **Note:**  
- Theoretical parts of assignments are sometimes answered in **English** and sometimes in **Persian**, depending on the nature of the question and convenience of explanation.  
- Practical parts include Python implementations, datasets, and results.  
- If you have any feedback, suggestions, or corrections, feel free to contribute or open an issue in the repository.  


---

##  Topics Covered  
- Introduction to Machine Learning  
- Linear Regression & Polynomial Regression  
- Logistic Regression & Classification  
- Bias-Variance Trade-off  
- Model Evaluation Metrics (MAE, MSE, RMSE, R², Accuracy, F1-score, etc.)  
- Regularization (Ridge, Lasso, ElasticNet)  
- Feature Scaling & Polynomial Features  
- Hypothesis Testing in ML  
- Loss Functions (MSE, MAE, Huber, Epsilon-sensitive)  
- Optimization & Gradient Descent Variants  
- Cross-validation, Grid Search, Random Search  
- Intro to Advanced Models (SVM, Decision Trees, Ensemble Methods)  

---
 

## Learning Outcomes
By the end of the course, you will be able to:
Explain core machine learning concepts and algorithms.
Build and evaluate ML models from scratch.
Apply preprocessing and feature engineering techniques.
Use Python ML libraries effectively for real-world datasets.
Compare models statistically to select the best-performing approac


## 📖 References
The Elements of Statistical Learning – Hastie, Tibshirani, Friedman
Pattern Recognition and Machine Learning – Bishop
Scikit-learn Documentation: https://scikit-learn.org/
UCI Machine Learning Repository: https://archive.ics.uci.edu/
