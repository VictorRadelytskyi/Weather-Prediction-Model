from sklearn.ensemble import RandomForestClassifier
from numpy import ndarray
import pickle
import os

def train(features_encoded_train, target_encoded_train, return_predictions: bool = True, features_encoded_test = None, savemodel=True, filename: str = "RandomForest_model.pkl") -> RandomForestClassifier | tuple[RandomForestClassifier, ndarray]:
    model = RandomForestClassifier()
    model.fit(features_encoded_train, target_encoded_train)

    if savemodel:
        output_path = os.path.join(os.path.dirname(__file__), "saved", "models", filename)
        models_dir = os.path.dirname(output_path) 
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(f'Can\'t save model, because {models_dir} is not a valid directory. Maybe {output_path} is not valid path?')
        
        with open(output_path, mode = 'wb') as file:
            pickle.dump(model, file)
    if not return_predictions:
        return model
    else:
        if features_encoded_test is None:
            raise ValueError("return_predictions flag is set to True, but no test features are provided")
        predictions = model.predict(features_encoded_test)
        return model, predictions


