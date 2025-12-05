import sys
import pickle
import numpy as np
from mycsp import MyCSP
from processor import EEGTraitement
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class LogReg:

    def __init__(self):
        [...]

    def ft_train_model(self, subject_id : int, run : int):
        processor = EEGTraitement(subject_id=subject_id, run=run)
        epochs = processor.ft_create_epochs(id_event=[1, 2], tmin=-0.5, tmax=4.0)

        csp = MyCSP(epochs=epochs)
        cov_t1, cov_t2 = csp.ft_compute_covariance_matrices()
        eigvals_sorted, eigvecs_sorted = csp.ft_eigenvalue_problem_covariance(cov_t1, cov_t2)
        W = csp.ft_compute_W_matrix(eigvals_sorted, eigvecs_sorted)
        features = csp.ft_obtain_csp_signals(W, csp.X)
        labels = csp.ft_return_Y_labels()

        # model = LogisticRegression(max_iter=5000, solver='lbfgs')
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(
                max_iter=1000, 
                solver='lbfgs',
            ))
        ])
        model.fit(features, labels)

        pred = model.predict(features)
        accuracy = np.mean(pred == labels)
        print(f"Accuracy: {accuracy}")
        return W, model


    def ft_predict_model(self, subject_id : int, run : int, type_run : str):
        model : LogisticRegression = None
        W : np.ndarray = None
        W, model = self.ft_load_model(f"./models/{type_run}.pkl")

        processor = EEGTraitement(subject_id=subject_id, run=run)
        epochs = processor.ft_create_epochs(id_event=[1, 2], tmin=-0.5, tmax=4.0)
        print(epochs.events)

        X_test = epochs.get_data()
        features = MyCSP(epochs=epochs).ft_obtain_csp_signals(W, X_test)
        y_pred = model.predict(features)

        return y_pred

    def ft_save_model(self, W : np.ndarray, model : LogisticRegression, type_run : str):
        try :   
            data = {
                "W": W,
                "model": model
            }
            with open(f"./models/{type_run}.pkl", "wb") as f:
                pickle.dump(data, f)

        except Exception as e:
            print(f"Error saving model: {e}")
            sys.exit(1)
    
    def ft_load_model(self, path : str):
        try :
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data["W"], data["model"]
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

def main():

    pipeline = LogReg()
    W, model = pipeline.ft_train_model(1, 6)
    pipeline.ft_save_model(W, model, "hands_right_left")

    W, model = pipeline.ft_load_model("./models/hands_right_left.pkl")
    y_pred = pipeline.ft_predict_model(1, 13, "hands_right_left")
    print(y_pred)



if __name__ == "__main__":
    main()