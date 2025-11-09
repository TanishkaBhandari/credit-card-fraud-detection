#  **💳 Credit Card Fraud Detection Using Machine Learning**



### ***## 📘 Overview***

This project focuses on detecting fraudulent credit card transactions using \*\*Machine Learning\*\* models and deploying it through a Flask web application for real-time predictions.



With the rapid growth of online payments, detecting fraudulent transactions has become critical.  

This project implements a trained ML model that identifies fraudulent activities based on transaction features.





### ***## 🧠 Key Features***

\- Data preprocessing, scaling, and model training  

\- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC  

\- Flask web interface for real-time transaction testing  

\- Saved trained model (`model.pkl`) for quick reuse  

\- Visualization using confusion matrix and performance metrics  



---



### ***## ⚙️ Files Overview***



`fraud\_detection.py` | Main file that trains and evaluates ML models (Logistic Regression, Decision Tree, Random Forest) |

`predict.py` | Loads trained model and predicts transaction type (Fraud / Genuine) |

`app.py` | Flask web app handling frontend and backend |

`model.pkl` | Trained model saved using pickle |

`requirements.txt` | Python dependencies |

| `/screenshots/` | Output and UI screenshots |



---



### ***## 🧰 Technologies Used***

\- Python

-Seaborn

-sklearn

\- Scikit-learn

\- Flask

\- Pandas, NumPy

\- Matplotlib



---



### ***## 📊 Model Performance***



| Accuracy | 93–99% |

| Precision | 0.0136 |

| Recall | 0.8333 |

| F1-Score | 0.0268 |

| ROC-AUC | 0.8849 |



---



### ***## 🚀 How to Run Locally***



1\. \*\*Clone this repository\*\*

&nbsp;  ```bash

&nbsp;  git clone https://github.com/yourusername/credit-card-fraud-detection.git

&nbsp;  cd credit-card-fraud-detection



2\. \*\*Install dependencies

pip install -r requirements.txt



3\. Run the flask app

python app.py



4\. Go to browser

http://127.0.0.1:5000



You can also check the screenshots folder in this repo for visual outputs.



### ***Project Workflow***



1. Data Cleaning \& Preprocessing
   
2. Feature Scaling
   
3. Model Training (Logistic Regression, Decision Tree, Random Forest)
   
4. Evaluation with Confusion Matrix and ROC-AUC
   
5. Saving Trained Model (model.pkl)
   
6. Integrating Flask frontend with backend prediction logic



### ***⭐ Future Improvements***



Add more ML models (XGBoost, SVM)



Enhance frontend with better UI



Integrate real-time data API for live fraud detection

