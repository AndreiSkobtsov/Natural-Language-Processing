import pandas as pd
import shap
import matplotlib.pyplot as plt

def shap_explain(classifier, X_data: pd.DataFrame, feature_names: list):
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(classifier.clf)
    shap_values = explainer.shap_values(X_data)
    
    plt.figure(figsize=(12, 8))
    
    # plot_type="bar" creates a clean stacked bar chart with a legend for multi-class
    shap.summary_plot(
        shap_values, 
        X_data, 
        feature_names=feature_names, 
        class_names=classifier.classes_, 
        plot_type="bar", 
        show=False
    )
    
    plt.suptitle("SHAP Feature Importance by Model", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()