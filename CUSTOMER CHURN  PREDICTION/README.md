# Customer Churn Prediction

## Project Overview

This project aims to predict customer churn for a subscription-based service or business. Customer churn refers to the rate at which customers stop using a service or subscription. Accurately predicting churn can help businesses retain customers and reduce revenue loss. This project uses historical customer data, including usage behavior and demographics, to build a predictive model.

## Dataset

The dataset includes the following features:

- **RowNumber**: Unique identifier for each row.
- **CustomerId**: Unique identifier for each customer.
- **Surname**: Customer's surname.
- **CreditScore**: Customer's credit score.
- **Geography**: Customer's country of residence.
- **Gender**: Customer's gender.
- **Age**: Customer's age.
- **Tenure**: Number of years the customer has been with the service.
- **Balance**: Account balance.
- **NumOfProducts**: Number of products the customer is using.
- **HasCrCard**: Indicates whether the customer has a credit card (1: Yes, 0: No).
- **IsActiveMember**: Indicates whether the customer is an active member (1: Yes, 0: No).
- **EstimatedSalary**: Estimated annual salary.
- **Exited**: Indicates whether the customer churned (1: Yes, 0: No).

## Project Steps

1. **Data Preprocessing**:
   - Handling missing values (if any).
   - Encoding categorical variables (e.g., Geography, Gender).
   - Feature scaling for numerical variables.
   - Splitting the dataset into training and testing sets.

2. **Modeling**:
   - Logistic Regression was used as the primary model to predict customer churn.
   - The model was trained on the historical customer data and tested on a separate test set to evaluate performance.

3. **Evaluation**:
   - The model's performance was evaluated using metrics such as Accuracy, Precision, Recall, and the F1 Score.
   - A confusion matrix was also used to visualize the performance of the model.

4. **Results**:
   - The Logistic Regression model was able to predict customer churn with [insert accuracy]% accuracy.
   - [Mention any insights or key findings from the model performance].


## Usage

The project can be used to predict customer churn for similar subscription-based services. The Logistic Regression model can be retrained with new data to improve predictions or adapted to different datasets.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to improve the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


