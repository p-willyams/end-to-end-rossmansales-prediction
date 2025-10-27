
<img width="1920" height="1080" alt="Análise(2)" src="https://github.com/user-attachments/assets/469b45f8-a791-46ed-bf92-6bd0e6b0977c" />

# End-To-End-Rossman-Sales-Prediction

## 1. Objective  
The goal of this project is to develop a predictive model capable of estimating store sales for the next six weeks for each pharmacy in a retail chain. This forecast enables the company owner to identify which stores have the greatest return potential and strategically plan investments, renovations, and marketing campaigns.

The model was built using historical sales data, store characteristics, promotions, and temporal information, applying Data Science and Machine Learning techniques to achieve accurate and actionable predictions.

## 2. Project Overview  
This project applies data science techniques, including data cleaning, exploratory analysis, and machine learning, to predict sales patterns and forecast store performance over the next six weeks.
The analysis aims to support business decisions, optimize resource allocation, and improve strategic planning across the pharmacy network.

## 3. Main Steps  
- **Data Description**: understanding variables and their relationships with sales behavior, and, data cleaning.
- **Feature Engineering**: creation of new variables based on dates, promotions, and store characteristics.
- **Exploratory Data Analysis (EDA)**: univariate, bivariate, and multivariate analysis to identify relevant patterns and correlations.
- **Data Preparation**: applying encodings and scaling to make the data suitable for machine learning algorithms.
- **Feature Selection**: selecting the most relevant attributes for model performance.
- **Machine Learning Modeling**: building and evaluating multiple regression algorithms.
- **Fine Tuning**: hyperparameter optimization to maximize model performance.
- **Error Translation and Interpretation**: analyzing the model’s results and understanding prediction errors in real-world scenarios.
- **Model Deployment**: creating a data pipeline and API for automated forecast generation.

## 4. Dataset  
The dataset used in this project contains daily sales records from the Rossmann pharmacy chain, which operates more than 3,000 stores across seven European countries. **[Rossman Sales Dataset](https://www.kaggle.com/competitions/rossmann-store-sales)**

Currently, each store manager is responsible for predicting their store’s sales for the next six weeks, a challenging task influenced by multiple factors such as promotions, competition, school holidays, seasonality, and location.

This variability makes manual forecasting inconsistent, which justifies the use of Data Science and Machine Learning techniques to produce more robust, data-driven predictions that support strategic business decisions.

## 5. Conclusion 
The project aimed to build a predictive model capable of estimating store sales for the next six weeks for the Rossmann pharmacy chain, supporting management in making strategic decisions regarding investments, promotions, and expansion.

Throughout the process, several key steps were carried out — including exploratory analysis, feature engineering, feature selection, model training and validation, and finally, production deployment through a data pipeline and an API for automated predictions.

Among the tested models, the XGBoost Regressor demonstrated the best balance between accuracy, computational efficiency, and generalization capability, and was selected for production. Its consistent performance showed that it is possible to reliably predict sales volume, even considering variations due to seasonality, promotions, and special dates.

Deploying the model made the system automated and scalable, allowing new predictions to be continuously generated as fresh data becomes available, with no manual intervention required.

This project demonstrates a practical application of Data Science and Machine Learning in a real-world business context, providing a solution that delivers strategic value, turning historical data into actionable insights and data-driven decisions. 
