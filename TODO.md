# Model and Prediction Enhancements TODO

## Plan Summary
- Enhance model testing with accuracy and calibration.
- Add batch predictions and file upload.
- Display detailed confidence and risk factors.
- Integrate explainability (SHAP/feature importance).

## Steps
1. [x] Enhance train_model.py: Add model evaluation (accuracy, calibration curves) and save metrics.
2. [x] Update app.py: Add batch prediction endpoint and detailed confidence/risk factors.
3. [x] Update app_streamlit.py: Add file upload for batch predictions, display SHAP values/feature importance, and more confidence details.
4. [ ] Create requirements.txt: Add dependencies like SHAP.
5. [ ] Create utils.py: For explainability utilities.
6. [ ] Install dependencies and test.
