# Data Science Portfolio by Sumit Watkar

## [Project 1: Credit Risk Modelling using Machine Learning](https://github.com/sumitwatkar/Credit-Risk-Modelling-using-Machine-Learning)

### Overview:

- The project focused on developing a reliable credit risk modeling system using advanced machine learning techniques to optimize the bank's lending strategy. The primary goals is to identify patterns that indicate if a person is unlikely to repay a loan (i.e., labeled as a bad risk) and to implement machine learning algorithms to build predictive models for loan risk assessment.

### Methodologies:

**1. Data Exploration and Preprocessing**: Examined the dataset and performed data cleaning, transformation, and feature engineering for model input.

**2. Model Development**: Developed and evaluated multiple machine learning models, including Logistic Regression, Decision Trees, Random Forest, Extra Trees, and XGBoost Classifier.

**3. Model Evaluation**: Used evaluation metrics such as accuracy, precision, recall, and ROC-AUC score to assess model performance.

**4. Model Optimization**: Tuned hyperparameters for each model to improve performance and generalization.

**5. Feature Importance Analysis**: Identified the top features impacting loan risk, such as the total principal received, last payment amount, and total payment.

### Technologies and Tools Used:

**1. Packages**: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn.

**2. Machine Learning Algorithms**: Logistic Regression, Decision Trees, Random Forest, Extra Trees, XGBoost.

### Outcomes:

**1. Best Model Selection**: Concluded that the XGBoost Classifier was the best model for predicting loan risk due to its high ROC-AUC score compared to other models.

**2. Key Insights**: Discovered that the top factors contributing to loan risk were the total principal received, last payment amount, and total payment.

**3. Actionable Recommendations**: Suggested actions such as rejecting loans, reducing loan amounts, or lending at higher interest rates based on risk assessment.

**4. Impact**: The project optimized the bank's lending strategy by providing a robust system for automatic loan application evaluation, thereby reducing potential losses and improving decision-making.


## [Project 2: Customer Segmentation using RFM Analysis](https://github.com/sumitwatkar/Customer-Segmentation-using-RFM-Analysis)

### Overview:

- The project aimed to analyze customer data and develop customer segmentation based on purchasing power and income levels using RFM analysis and machine learning models. This approach enabled the business to understand the spending behavior and purchasing power of different customer segments, thereby optimizing marketing efforts and increasing revenue. The project utilized clustering algorithms such as K-means Clustering,  Gaussian Mixture Model (GMM) for segmentation, as well as a comprehensive data pipeline for preprocessing, validation, transformation, and model training.

### Methodologies:

**1. Data Ingestion and Validation**: Ingested customer data and performed data validation checks, including file name, column labels, data types, and missing values validation.

**2. Data Transformation and Feature Engineering**: Transformed and engineered features from the data to enhance model performance, including creating new features based on insights from exploratory data analysis (EDA).

**3. Model Training and Evaluation**: Utilized clustering algorithms like K-means Clustering and Gaussian Mixture Model (GMM) to segment customers based on income and purchasing power and evaluated models using metrics such as the silhouette score.

**4. Model Pushing**: Saved the optimal models and associated reports for future use, ensuring the best-performing models were available for batch predictions.

**5. UI for User Interaction**: Developed a user-friendly interface for stakeholders to interact with trained models and cluster labeling.

### Technologies and Tools Used:

**1. Packages**: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

**2. Machine Learning Algorithms**: Clustering algorithms such as K-means Clustering and Gaussian Mixture Model (GMM)

### Outcomes:

**1. Customer Segmentation**: Identified key customer segments such as "Affluent Customers" (high income, high spending), "Middle-Income Customers" (moderate income and spending), and "Budget-Conscious Customers" (low income, low spending).

**2. Enhanced Marketing Strategies**: Provided actionable insights for targeted marketing efforts, enabling the business to develop customized strategies for each customer segment.

**3. Optimized Data Pipeline**: Implemented a robust data pipeline that included data ingestion, validation, transformation, and model training, ensuring high data quality and consistency for model development.

**4. Model Selection and Deployment**: Trained and evaluated machine learning models, saving the best-performing models for future predictions and batch predictions.


## [Project 3: Text to Speech Generation using CICD](https://github.com/sumitwatkar/Text-to-Speech-Generation-using-CICD)

### Overview:

- The Text-to-Speech (TTS) project is a Python-based application that converts written text into clear and natural speech output. Supporting multiple languages, the application leverages the Google Text-to-Speech (GTTS) library to synthesize speech from text, making it suitable for a variety of use cases including assisting visually impaired users, language learners, and individuals who prefer to consume textual content in an auditory format.

- The project adopts continuous integration and continuous deployment (CICD) practices to streamline the development and deployment process, allowing for efficient testing and deployment of the application.

### Methodologies:

**1. Application Development**: Built the TTS application using Python and the GTTS library, optimizing the synthesis process for high-quality speech generation and implementing error handling for smooth user interactions.

**2. CICD Pipeline**: Integrated GitHub Actions for automated testing and deployment, including code linting, unit tests, and automated builds.

**3. Containerization**: Created Dockerfiles to containerize the application, ensuring consistent and efficient deployments.

**4. Infrastructure Management**: Managed deployment and infrastructure using AWS services, deploying Docker images to Amazon Elastic Container Registry (ECR) for seamless integration and hosting the application on AWS Amazon instances for scalability and flexibility.

### Technologies and Tools Used:

* **Libraries**: GTTS, numpy, pandas
* **Web Framework**: Flask
* **Version Control & CICD**: GitHub Actions
* **Containerization**: Docker
* **Deployment & Infrastructure**: Amazon EC2, Amazon ECR

### Outcomes:

**1. Streamlined Development**: The CICD pipeline automates code linting, testing, building, and deployment, offering a smooth development lifecycle.

**2. Efficient Deployment**: Automated deployment of the TTS application to Amazon EC2 instances ensures fast, consistent, and reliable deployments.

**3. Containerization**: Docker containers ensure consistent application behavior across different environments and easy deployment.

**4. Scalability**: Leverages AWS services for scalability and to handle real-time synthesis challenges.

**5. Enhanced Productivity**: CICD practices improve efficiency, reduce manual errors, and accelerate deployment cycles.


## [Project 4: Flight Price Prediction using Amazon SageMaker](https://github.com/sumitwatkar/Text-to-Speech-Generation-using-CICD)

### Overview:

- The project aims to develop and deploy machine learning models for predicting flight ticket prices using Amazon SageMaker. This involves building a robust model that can accurately forecast the cost of a flight based on the various input features. The project also leverages the power of Amazon SageMaker for efficient training, optimization of the machine learning models.

### Methodologies:

**1. Data Collection and Preprocessing**: Acquired and preprocessed extensive datasets containing historical flight information, such as prices, routes, departure and arrival times, and airlines.

**2. Exploratory Data Analysis (EDA)**: Conducted thorough EDA using pandas and numpy to uncover patterns, trends, and potential features for model development.

**3. Model Development and Fine-Tuning**: Created and optimized machine learning models, with a focus on the high-performance XGBoost algorithm for its efficiency and interpretability.

**4. Feature Engineering and Selection**: Engineered and selected impactful features from the dataset using `feature-engine` to enhance model accuracy and performance.

**5. Model Evaluation**: Assessed model performance using metrics such as mean absolute error (MAE), mean squared error (MSE), and R-squared to ensure reliability and precision.

**6. Model Deployment**: Deployed the prediction model using Streamlit, offering a user-friendly web interface where users could input flight details and receive price predictions.

### Technologies and Tools Used:

* **Libraries**: numpy, pandas, matplotlob, seaborn, scipy, feature-engine, scikit-learn
* **Machine Learning Algorithm**: Isolation Forest, Extreme Gradient Boosting (XG Boost)
* **Web Framework**: Streamlit
* **Deployment & Infrastructure**: Streamlit Cloud

### Outcomes:

**1. Accurate Price Predictions**: Developed a reliable machine learning model that predicts flight ticket prices accurately based on historical data and selected features.

**2. User-Friendly Interface**: Created a streamlined web interface using Streamlit, enabling users to input flight details and swiftly receive price predictions.

**3. Improved Model Performance**: Enhanced model accuracy through fine-tuning, feature engineering, and iterative development, leading to dependable and precise predictions.

**4. Insights into Flight Pricing**: Provided valuable insights into the factors affecting flight prices, facilitating better decision-making for travelers and airline companies.

*[Web App Link](https://flight-price-prediction-using-amazon-sagemaker.streamlit.app/)*
