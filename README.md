# Breast Cancer Diagnostic Classifiers

Five classification models score a held-out test set from the Breast Cancer Wisconsin (Diagnostic) data. The Streamlit app loads the saved models and reports metrics. It does not retrain.

## a. Problem statement

A diagnostic clinic wants to know whether a breast mass is malignant or benign using 30 numeric measurements taken from a fine-needle aspirate image of cell nuclei. The practical question is which of the five required classifiers is reliable enough on unseen patients, with extra weight on catching malignant cases (recall) without calling too many benign cases malignant (precision). Matthews correlation is included because the classes are uneven.

## b. Dataset description

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository, loaded with `sklearn.datasets.load_breast_cancer`
- **Instances:** 569 (above the 500 minimum)
- **Features:** 30 numeric features (above the 12 minimum). For each of ten nucleus measurements — radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension — the data records the mean, the standard error, and the worst (largest) value.
- **Target:** `diagnosis`. Malignant is encoded as **1** (212 rows). Benign is encoded as **0** (357 rows). scikit-learn’s original coding is the reverse; `train_models.py` flips it so the positive class is malignant.
- **Missing values:** none
- **Split used for the saved metrics:** stratified 80/20, `random_state=42`. Training rows: 455. Test rows: 114 (42 malignant, 72 benign). The test rows are stored in `test_data.csv`.
- **Metric definitions:** accuracy, ROC AUC, precision, recall, and F1 use malignant (`1`) as the positive class. AUC uses the predicted probability of malignant. MCC is computed on both classes.

## c. GitHub Repository Link

Replace this line with the public repository URL after you push:

`https://github.com/<your-username>/<your-repo>`

Live Streamlit app (fill in after deploy):

`https://<your-app>.streamlit.app`

## d. Models used

All five models were fit on the same training split. Logistic regression, k-nearest neighbors, and the support vector machine scale features inside a pipeline. The decision tree is limited to depth 5 and a minimum of 4 samples per leaf so it does not memorize the small training set. Naive Bayes is Gaussian. The SVM is an RBF `SVC` wrapped in `CalibratedClassifierCV` so it can produce probabilities for AUC.

Metrics below are on `test_data.csv` only.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.8772 | 0.9654 | 0.9118 | 0.7381 | 0.8158 | 0.7343 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong and stable. On 114 test rows it misses 3 of 42 malignant cases and flags 1 benign case. It has the highest AUC (0.9960), so its probability ranking is the best even though accuracy is just behind the SVM. A linear boundary fits these nucleus measurements well, and the model stays easy to explain. |
| Decision Tree | Weakest of the five. It misses 11 malignant cases (recall 0.7381) and is the only model with MCC below 0.80. Axis-aligned splits struggle because radius, perimeter, and area move together. The depth cap avoids a perfect training fit, and the test result shows the remaining tree still does not generalize as well as the other models. |
| KNN | Second tier, close to logistic regression. With k = 7 on scaled features it misses 4 malignant cases and flags 1 benign case. Scaling matters here: area and smoothness are on very different ranges, and unscaled distance would be dominated by area. AUC (0.9825) is the lowest after the tree. |
| Naive Bayes | Perfect precision on this split: every row it called malignant was malignant (0 false positives). It still misses 7 malignant cases, so recall is only 0.8333. AUC remains high (0.9934), which means the scores rank cases well and the 0.5 threshold is simply conservative. The Gaussian assumption is approximate because several features are skewed. |
| Support Vector Machine | Best accuracy (0.9737), best F1 (0.9639), best MCC (0.9433), and best malignant recall (0.9524, only 2 misses). The RBF kernel on scaled features separates the two classes with a wider margin than the tree or k-NN. AUC (0.9947) is just below logistic regression. |
| Overall, Winner for your dataset? | **Support Vector Machine.** It leads accuracy, precision-recall balance, F1, MCC, and malignant recall. Logistic regression is the runner-up and the better choice if a simple, inspectable score is preferred, because it has the highest AUC and only one extra missed malignant case. |

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

`train_models.py` rewrites `model/*.joblib`, `model/metadata.json`, and `test_data.csv`. The app only loads those files.

In the app, upload a CSV that contains the 30 feature columns plus `diagnosis` (0 or 1), or leave the bundled hold-out file selected. Choose a model from the dropdown. The page shows the six metrics, a confusion matrix, a classification report, and a comparison table for all five models. Uploaded rows are not used for training.

## Repository layout

```text
app.py
train_models.py
requirements.txt
README.md
test_data.csv
model/
  logistic_regression.joblib
  decision_tree.joblib
  knn.joblib
  naive_bayes.joblib
  svm.joblib
  metadata.json
```
