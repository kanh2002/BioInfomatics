import joblib
from xgboost import XGBClassifier
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Load các thành phần
vectorizer = joblib.load(MODEL_DIR / 'tfidf_vectorizer.joblib')
nusvc = joblib.load(MODEL_DIR / 'nusvc_diabetes_model.joblib')

# Load XGBoost
xgb = XGBClassifier()
xgb.load_model(str(MODEL_DIR / 'xgb_diabetes_model.json'))

# Ví dụ dự đoán một chuỗi DNA mới
new_seq = ["""TTTTTTTACCAATATAAACAGGGCCGTTGACCCTTTCATTTTATTAAAATGGCACATAATTATTAAAACA
GCATACTGATCACTTTATACTTCTGCTAGCCCCCAGGGGAGCTGCTGGGGGCGGCATGTGAGTGCCCTCC
CGAAGGGTACAGATTCATGCATTGAGCAATTCGTGTTCTTTATCGGTTTTCCCAACAGCATCAGGATTTG
AGAGTGGGTCGAGGTCAGCGAAGAGGCTGAACCAGGCAGTCAGGTCTGAGGCAGCCTTAGCAGGTTCTGG
GGAGAGAAGAGGAAACATGAGCAAACGCACCTTCCAAATGTCCACCTCTGCCATGCGGGATGCAGGCAGG
TCCAGGTCATC"""]   # chuỗi DNA của bạn
X_new = vectorizer.transform(new_seq)

pred_nu = nusvc.predict(X_new)
pred_xgb = xgb.predict(X_new)

print("pred_nu: ", pred_nu)
print("NuSVC dự đoán:", "Diabetic" if pred_nu[0] == 1 else "Non-diabetic")
print("XGBoost dự đoán:", "Diabetic" if pred_xgb[0] == 1 else "Non-diabetic")