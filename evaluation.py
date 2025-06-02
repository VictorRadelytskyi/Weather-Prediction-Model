from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def evaluate(predictions, target_encoded_test, label_encoder: LabelEncoder, display_accuracy: bool = True, display_class_report: bool = True) -> tuple[float, str | dict[Any, Any]]:

    accuracy = accuracy_score(target_encoded_test, predictions)
    class_report = classification_report(target_encoded_test, predictions, target_names = label_encoder.classes_)

    if display_accuracy:
        print(f'\nAccuracy: {accuracy}\n')

    if display_class_report:
        print(f'\nClassification report: {class_report}\n')

    return float(accuracy), class_report 

def display_confusion_matrix(predictions, target_encoded_test, label_encoder: LabelEncoder, save: bool = True, filename: str = "confusion_matrix.png") -> None:
    confusion_matrix_ = confusion_matrix(target_encoded_test, predictions)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_.tolist(), yticklabels=label_encoder.classes_.tolist())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if save:
        output_path = os.path.join(os.path.dirname(__file__), "saved", "img", filename)
        img_dir = os.path.dirname(output_path) 
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f'Can\'t save confusion matrix, because {img_dir} is not a valid directory. Maybe {output_path} is not valid path?')
        
        plt.savefig(output_path)

    plt.show()
    plt.close(fig)
    

def show_cross_validation(model: RandomForestClassifier, features_encoded, target_encoded) -> None:
    cv_scores = cross_val_score(model, features_encoded, target_encoded, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean()}")

def feature_importance(model, features_encoded, save: bool = True, filename: str = "feature_importance.png") -> None:
    importances = model.feature_importances_
    features_names = features_encoded.columns
    importance_df = pd.DataFrame({'Feature': features_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    fig = plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')

    if save:
        output_path = os.path.join(os.path.dirname(__file__), "saved", "img", filename)
        img_dir = os.path.dirname(output_path) 
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f'Can\'t save feature importance graph, because {img_dir} is not a valid directory. Maybe {output_path} is not valid path?')
        
        plt.savefig(output_path)
        
    plt.show()
    plt.close(fig)

