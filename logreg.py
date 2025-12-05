import sys
import time
import pickle
import numpy as np
from mycsp import MyCSP
from processor import EEGTraitement
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class LogReg:
    """
        Class for training and predicting motor imagery classification models
        using Common Spatial Patterns (CSP) and Logistic Regression.
    """

    def ft_identify_run_type(self, run: int):
        """
            Logic:
            - Identifies the motor imagery task type based on the run number
            - Maps run numbers to task types:
              left_fist_right_fist (runs 3,4,7,8,11,12)
              or both_fists_both_feet (runs 5,6,9,10,13,14)
            Return:
            - run_type (str): The task type identifier or "unknown" if run
              is not recognized
        """
        if run in [3, 4, 7, 8, 11, 12]:
            return "left_fist_right_fist"
        elif run in [5, 6, 9, 10, 13, 14]:
            return "both_fists_both_feet"
        else:
            return "unknown"

    def ft_train_model(self, subject_id: int, run: int, print_mode: bool):
        """
            Logic:
            - Identifies the run type for the given run number
            - Calls ft_pipeline_train to process EEG data and train the model
            - Saves the trained model and CSP transformation matrix to disk
            Return:
            - None
        """
        run_type = self.ft_identify_run_type(run)
        if run_type is None:
            print(f"Run {run} not found")
            sys.exit(1)

        W, model = self.ft_pipeline_train(subject_id, run, print_mode)
        self.ft_save_model(W, model, run_type, print_mode)

    def ft_pipeline_train(self, subject_id: int, run: int, print_mode: bool):
        """
            Logic:
            - Loads EEG data for the specified subject and run
            - Creates epochs from the raw EEG data
            - Computes CSP transformation matrix (W) from covariance matrices
            - Extracts CSP features from the EEG epochs
            - Performs cross-validation to evaluate the model
            - Trains a Logistic Regression classifier on the features
            Return:
            - W (np.ndarray): CSP transformation matrix
            - model (LogisticRegression): Trained classifier
        """
        processor = EEGTraitement(subject_id=subject_id, run=run)
        epochs = processor.ft_create_epochs(
            id_event=[1, 2], tmin=-0.5, tmax=4.0)

        csp = MyCSP(epochs=epochs)
        cov_t1, cov_t2 = csp.ft_compute_covariance_matrices()
        eigvals_sorted, eigvecs_sorted = csp.ft_eigenvalue_problem_covariance(
            cov_t1, cov_t2)
        W = csp.ft_compute_W_matrix(eigvals_sorted, eigvecs_sorted)
        features = csp.ft_obtain_csp_signals(W, csp.X)
        labels = csp.ft_return_Y_labels()

        model = LogisticRegression(max_iter=10000)

        min_class_samples = np.min(np.unique(labels, return_counts=True)[1])
        n_folds = min(10, min_class_samples)
        scores = cross_val_score(model, features, labels, cv=n_folds)
        if print_mode:
            print(np.round(scores, 4))
            print(f"cross_val_score: {scores.mean():.4f}")
        model.fit(features, labels)
        return W, model

    def ft_predict_model(
            self,
            subject_id: int,
            run: int,
            print_mode: str,
            stream_mode: bool = False):
        """
            Logic:
            - Identifies the run type for the given run number
            - Routes to stream or batch prediction mode based on stream_mode
            - In stream mode: processes epochs one by one with timing info
            - In batch mode: processes all epochs at once and calculates
              accuracy
            Return:
            - y_pred: Prediction accuracy (float) or None in stream mode
        """
        run_type = self.ft_identify_run_type(run)
        if run_type is None:
            print(f"Run {run} not found")
            sys.exit(1)

        if stream_mode:
            return self.ft_pipeline_predict_stream(subject_id, run, run_type)
        else:
            y_pred = self.ft_pipeline_predict(
                subject_id, run, run_type, print_mode)
        return y_pred

    def ft_accuracy_calculation(
            self,
            original_labels: list[int],
            y_pred: list[int],
            print_mode: bool):
        """
            Logic:
            - Compares predicted labels with original true labels
            - Counts the number of correct predictions
            - Optionally prints detailed comparison for each epoch
            - Calculates accuracy as correct predictions / total predictions
            Return:
            - accuracy_total (float): The accuracy score between 0.0 and 1.0
        """
        total = min(len(original_labels), len(y_pred))
        correct = 0
        if print_mode:
            print("epoch nb: [prediction] [truth] equal?")
        for i in range(total):
            if original_labels[i] == y_pred[i]:
                correct += 1
            if print_mode:
                print(
                    f"epoch {
                        i:02d}:         [{
                        y_pred[i]:02d}]    [{
                        original_labels[i]:02d}] {
                        original_labels[i] == y_pred[i]}")

        accuracy_total = correct / total
        if print_mode:
            print(f"Accuracy: {accuracy_total}")

        return accuracy_total

    def ft_pipeline_predict_stream(
            self,
            subject_id: int,
            run: int,
            type_run: str):
        """
            Logic:
            - Loads the trained model and CSP transformation matrix
            - Loads EEG data for the specified subject and run
            - Processes each epoch individually in a loop
            - Extracts CSP features for each epoch
            - Predicts the class for each epoch and measures processing time
            - Prints prediction, truth label, and processing time for each
              chunk
            Return:
            - None
        """
        model: LogisticRegression = None
        W: np.ndarray = None
        W, model = self.ft_load_model(f"./models/{type_run}.pkl")

        processor = EEGTraitement(subject_id=subject_id, run=run)
        epochs = processor.ft_create_epochs(
            id_event=[1, 2], tmin=-0.5, tmax=4.0)

        X = epochs.get_data()
        y = epochs.events[:, 2]

        csp = MyCSP(epochs=epochs)

        for i, epoch in enumerate(X):
            t0 = time.time()
            features = csp.ft_obtain_csp_signals(W, epoch[None, ...])
            pred = model.predict(features)[0]
            dt = time.time() - t0
            print(f"chunk {i:02d}: pred={pred}, truth={y[i]}, time={dt:.6f}s")

        return None

    def ft_pipeline_predict(
            self,
            subject_id: int,
            run: int,
            type_run: str,
            print_mode: str):
        """
            Logic:
            - Loads the trained model and CSP transformation matrix from disk
            - Loads EEG test data for the specified subject and run
            - Extracts CSP features from all test epochs
            - Makes predictions for all epochs at once
            - Calculates and returns accuracy based on print_mode:
              - "full": prints detailed per-epoch comparison
              - "summary": returns accuracy without detailed output
            Return:
            - accuracy (float): The model accuracy on the test data
        """
        model: LogisticRegression = None
        W: np.ndarray = None
        W, model = self.ft_load_model(f"./models/{type_run}.pkl")

        processor = EEGTraitement(subject_id=subject_id, run=run)
        epochs = processor.ft_create_epochs(
            id_event=[1, 2], tmin=-0.5, tmax=4.0)

        original_labels = epochs.events[:, 2]
        X_test = epochs.get_data()
        features = MyCSP(epochs=epochs).ft_obtain_csp_signals(W, X_test)
        y_pred = model.predict(features)

        if print_mode == "full":
            return self.ft_accuracy_calculation(original_labels, y_pred, True)
        elif print_mode == "summary":
            return self.ft_accuracy_calculation(original_labels, y_pred, False)
        else:
            print(f"Invalid print mode: {print_mode}")
            sys.exit(1)

    def ft_save_model(
            self,
            W: np.ndarray,
            model: LogisticRegression,
            type_run: str,
            print_mode: bool):
        """
            Logic:
            - Creates a dictionary containing the CSP transformation matrix
              (W) and trained model
            - Saves the dictionary to a pickle file in the ./models/
              directory
            - Filename is based on the run type
              (e.g., "left_fist_right_fist.pkl")
            - Optionally prints a confirmation message if print_mode is True
            Return:
            - None
        """
        try:
            data = {
                "W": W,
                "model": model
            }
            with open(f"./models/{type_run}.pkl", "wb") as f:
                pickle.dump(data, f)

            if print_mode:
                print(f"Model saved in ./models/{type_run}.pkl")

        except Exception as e:
            print(f"Error saving model: {e}")
            sys.exit(1)

    def ft_load_model(self, path: str):
        """
            Logic:
            - Loads a saved model dictionary from a pickle file
            - Extracts the CSP transformation matrix (W) and trained model
            - Handles file not found or corruption errors with helpful messages
            Return:
            - W (np.ndarray): CSP transformation matrix
            - model (LogisticRegression): Trained classifier
        """
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data["W"], data["model"]
        except Exception as e:
            print(
                "Be sure that the model exists ! "
                "or train the model before predicting !")
            print("Run types: ")
            print("left_fist_right_fist : runs 3,4,7,8,11,12")
            print("both_fists_both_feet : runs 5,6,9,10,13,14")
            print(f"Error loading model: {e}")
            sys.exit(1)


def main():

    pipeline = LogReg()
    pipeline.ft_train_model(1, 6)
    y_pred = pipeline.ft_predict_model(1, 13)
    print(y_pred)


if __name__ == "__main__":
    main()
