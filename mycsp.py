import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Optional
from processor import EEGTraitement
import mne
import matplotlib.pyplot as plt
plt.style.use('./vortex.mplstyle')

class MyCSP:
    def __init__(self, epochs: mne.Epochs):
        self.epochs = epochs
        # shape: (n_epochs, n_channels, n_times)
        self.X = self.epochs.get_data()
        # shape: (n_epochs,)
        self.Y = self.epochs.events[:, -1]
        print(self.X.shape)
        print(self.Y.shape)

    # Equation 3 page 4
    def ft_compute_covariance_matrices(self):
        
        cov_pos = []
        cov_neg = []

        for i in range(self.X.shape[0]):
            if self.Y[i] == 2:
                result_positive = self.X[i] @ self.X[i].T
                # flat signal covariance matrix
                if np.trace(result_positive) < 1e-12:
                    continue
                result_pos_norm = result_positive / np.trace(result_positive)
                cov_pos.append(result_pos_norm)

            elif self.Y[i] == 3:
                result_negative = self.X[i] @ self.X[i].T
                # flat signal covariance matrix
                if np.trace(result_negative) < 1e-12:
                    continue
                result_neg_norm = result_negative / np.trace(result_negative)
                cov_neg.append(result_neg_norm)

        # Equation 3 page 4
        cov_pos = np.mean(cov_pos, axis=0)
        cov_neg = np.mean(cov_neg, axis=0)
        # Return np matrix
        return np.array(cov_pos), np.array(cov_neg)

    # Equation 5 page 4
    def ft_eigenvalue_problem_covariance(self, cov_t1 : np.ndarray, cov_t2 : np.ndarray):

        eigvals, eigvecs = scipy.linalg.eig(cov_t1, cov_t2)
        index_vals = np.argsort(eigvals)
        eigvals_sorted = eigvals[index_vals]
        eigvecs_sorted = eigvecs[:, index_vals]
        # self.ft_plot_discriminative_model(eigvals, eigvecs)
        # self.ft_plot_discriminative_model(eigvals_sorted, eigvecs_sorted)

        return eigvals_sorted, eigvecs_sorted
        

    def ft_plot_discriminative_model(self, eigvals_sorted : np.ndarray, eigvecs_sorted : np.ndarray):
        v_min = eigvecs_sorted[:, 0]
        v_max = eigvecs_sorted[:, -1]

        var_min = []
        var_max = []
        for i in range(self.X.shape[0]):
            X_i = self.X[i]
            z_min = v_min @ X_i
            z_max = v_max @ X_i
            var_min.append(np.var(z_min))
            var_max.append(np.var(z_max))

        # Affichage avec des couleurs personnalisées : bleu pour Y==2 et rouge pour Y==3
        colors = ['blue' if y == 2 else 'red' for y in self.Y]
        plt.scatter(var_min, var_max, c=colors, marker='o', alpha=0.7)
        plt.xlabel("Variance sur filtre v_min")
        plt.ylabel("Variance sur filtre v_max")

        plt.show()


    def ft_compute_W_matrix(self, eigvals_sorted : np.ndarray, eigvecs_sorted : np.ndarray):
        k = 3
        eigvecs_sorted = eigvecs_sorted.real
        W_small = eigvecs_sorted[:, :k].T
        W_large = eigvecs_sorted[:, -k:].T
        W = np.vstack([
            W_small, 
            W_large
        ])
        print(W.shape)
        return W

    def ft_obtain_csp_signals(self, W : np.ndarray, X : np.ndarray):
        features = []
        for i in range(X.shape[0]):
            X_i = X[i]
            z_i = W @ X_i
            f_i = np.log(np.var(z_i, axis=1))
            features.append(f_i)
        features = np.array(features) 
        return features


    def ft_return_Y_labels(self):
        return self.Y

def main():
    processor = EEGTraitement(subject_id=1, run=4)
    # processor.ft_plot_data(processor.raw_data, title='Raw EEG Data')
    
    epochs = processor.ft_create_epochs(id_event=[1, 2], tmin=-0.5, tmax=4.0)
    # processor.ft_plot_epochs(epochs)
    print(epochs.event_id)

    csp = MyCSP(epochs=epochs)

    cov_t1, cov_t2 = csp.ft_compute_covariance_matrices()
    # print(cov_t1)
    # print(cov_t2)

    eigvals_sorted, eigvecs_sorted = csp.ft_eigenvalue_problem_covariance(cov_t1, cov_t2)
    
    W = csp.ft_compute_W_matrix(eigvals_sorted, eigvecs_sorted)

    features = csp.ft_obtain_csp_signals(W, csp.X)
    print(features)


if __name__ == "__main__":
    main()