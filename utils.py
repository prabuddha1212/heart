import shap
import pandas as pd


def get_shap_values(model, input_df: pd.DataFrame):
    """
    Computes SHAP values for a given model and input data.

    Args:
        model: The trained scikit-learn model.
        input_df (pd.DataFrame): A single row DataFrame containing the feature data for prediction.

    Returns:
        tuple: A tuple containing the SHAP explainer and the SHAP explanation object.
    """
    try:
        # Create a SHAP explainer for the model.
        explainer = shap.Explainer(model, input_df)

        # Compute the SHAP values for the input data.
        shap_values = explainer(input_df)

        return explainer, shap_values
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        return None, None


def plot_shap_waterfall(shap_explanation, max_display=10):
    """
    Generates a SHAP waterfall plot to explain a single prediction.

    Args:
        shap_explanation: The SHAP Explanation object for a single instance.
        max_display (int): The maximum number of features to display in the plot.
    """
    shap.waterfall_plot(shap_explanation[0], max_display=max_display, show=False)
