# PRODIGY_DS_03 - Decision Tree Classifier for Bank Marketing

This task involves building a decision tree classifier to predict if a client will subscribe to a term deposit based on demographic and behavioral data.

## Dataset
The dataset used is the [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) from the UCI Machine Learning Repository.

## Methodology
1. **Preprocessing**: Handled categorical variables and split the data into training and testing sets.
2. **Modeling**: Built a Decision Tree Classifier using `sklearn`.
3. **Evaluation**: Evaluated the model using a confusion matrix and visualized the tree.

## Project Structure
- `preprocess.py`: Data cleaning and feature engineering.
- `data_loader.py`: Loading the bank marketing data.
- `model.py`: Training and evaluating the Decision Tree Classifier.
- `decision_tree.png`: Visualization of the trained decision tree.
- `confusion_matrix.png`: Model performance summary.

## How to use
Run the `model.py` script to train and evaluate the model.
```bash
python model.py
```
