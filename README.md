Credit Card Fraud Detection

This project uses machine learning to detect fraudulent credit card transactions and classify fraud types. It also explores how transaction patterns and model performance change over time.

## Project overview

- Explores 120,000 transactions from January to June 2026. About 9.49% are labeled as fraud.
- Compares Logistic Regression, Decision Tree, and Random Forest for binary fraud detection.
- Tests whether SMOTENC improves Logistic Regression on imbalanced data.
- Uses a separate Random Forest to classify transactions as legitimate or one of five fraud types.

## Results

The models trained on January–February transactions. March was used for validation and model selection. April–June was used for testing.

| Binary model | March fraud precision | March fraud recall | March fraud F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.9850 | 0.9675 | 0.9762 |
| Decision Tree | 0.9974 | 0.9932 | **0.9953** |
| Random Forest | 0.9995 | 0.9858 | 0.9926 |
| Logistic Regression + SMOTENC | 0.9345 | 0.9738 | 0.9537 |

The Decision Tree had the highest validation F1. On the April–June test set, it caught **5,718** fraud cases, missed **47**, and incorrectly flagged **15** legitimate transactions. SMOTENC increased Logistic Regression's recall, while the additional false alarms reduced its F1.

For fraud type prediction, the multiclass Random Forest's macro F1 fell from **0.947** in March to **0.913** in April–June. Synthetic-Identity was the hardest class to identify, with **61.6% recall** in the test period.

## Run the notebook

1. Obtain authorized access to `credit_card_fraud_drift_2026.csv` and place it beside `fraud_detection.ipynb`. The dataset is distributed separately.
2. Install the Python packages in the environment you use for Jupyter:

   ```bash
   python -m pip install pandas matplotlib scikit-learn imbalanced-learn joblib ipykernel
   ```

3. Open `fraud_detection.ipynb` in VS Code with the Python and Jupyter extensions. Select the same Python environment as the notebook kernel, then click **Run All**.
4. The notebook writes evaluation tables and fitted pipelines to the local `results/` folder.

## Limitations

The results describe the supplied dataset. Several historical features were strong predictors, but their calculation timing was not independently verified. For use on new transactions, those features should use only information available at prediction time. The changes observed across months do not establish what caused the decline in fraud type classification.
