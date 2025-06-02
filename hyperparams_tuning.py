from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import os
import pickle


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

def find_best_params(features_encoded_train, target_encoded_train, display_results: bool = True, savemodel: bool = True, filename: str = "GridSearch.pkl") -> GridSearchCV:
    grid_search = GridSearchCV(RandomForestClassifier(random_state=10), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(features_encoded_train, target_encoded_train)
    
    if display_results:
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_}")

    if savemodel:
        output_path = os.path.join(os.path.dirname(__file__), "saved", "models", filename)
        model_dir = os.path.dirname(output_path)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f'Can\'t save grid search model, because {model_dir} is not a valid directory. Maybe {output_path} is not valid path?')
        
        with open(output_path, mode="wb") as file:
            pickle.dump(grid_search, file)

    return grid_search
    