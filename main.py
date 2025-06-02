import time
from . import preprocessing, model_training, evaluation, hyperparams_tuning

def run() -> None:
    start_time: float = time.time()

    features, target = preprocessing.preprocess_data(show_data=True)
    features_encoded, target_encoded, label_encoder = preprocessing.encode_data(features, target)
    features_encoded_train, features_encoded_test, target_encoded_train, target_encoded_test = preprocessing.split(features_encoded, target_encoded)

    model, predictions = model_training.train(features_encoded_train, target_encoded_train, return_predictions = True, features_encoded_test = features_encoded_test)

    evaluation.evaluate(predictions, target_encoded_test, label_encoder)
    evaluation.display_confusion_matrix(predictions, target_encoded_test, label_encoder)
    evaluation.show_cross_validation(model, features_encoded, target_encoded)
    evaluation.feature_importance(model, features_encoded)

    hyperparams_tuning.find_best_params(features_encoded_train, target_encoded_train)

    print(f'The code has been executed in {time.time() - start_time} seconds')

if __name__ == "__main__":
    run()
