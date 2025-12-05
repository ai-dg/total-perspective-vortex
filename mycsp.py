import mne
import scipy
import numpy as np
from typing import Optional
from processor import EEGTraitement
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
plt.style.use('./vortex.mplstyle')


class MyCSP(BaseEstimator, TransformerMixin):
    """
        Class for implementing Common Spatial Patterns (CSP) algorithm.
        Computes CSP transformation matrix and extracts discriminative features
        from EEG epochs for motor imagery classification.
    """

    def __init__(self, epochs=None, n_components: int = 6):
        """
            Logic:
            - Initializes CSP transformer for use in Pipeline
            - Stores epochs if provided (for backward compatibility)
            - Sets number of CSP components
            Return:
            - None
        """
        self.epochs = epochs
        self.n_components = n_components
        self.W = None
        if epochs is not None:
            self.X = self.epochs.get_data()
            self.Y = self.epochs.events[:, -1]
        else:
            self.X = None
            self.Y = None

    # ************ Equation 3 page 4 ****************** #

    def ft_compute_covariance_matrices(self):
        """
            Logic:
            - Computes normalized covariance matrices for each epoch
            - Separates epochs by class: label 2 (positive) and label 3
              (negative)
            - Filters out epochs with trace < 1e-12 to avoid numerical issues
            - Normalizes each covariance matrix by its trace
            - Averages covariance matrices within each class (Equation 3)
            Return:
            - cov_pos (np.ndarray): Average normalized covariance matrix
              for class 2
            - cov_neg (np.ndarray): Average normalized covariance matrix
              for class 3
        """
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

        # ************ Equation 3 page 4 ****************** #
        cov_pos = np.mean(cov_pos, axis=0)
        cov_neg = np.mean(cov_neg, axis=0)
        # Return np matrix
        return np.array(cov_pos), np.array(cov_neg)

    # ************ Equation 5 page 4 ****************** #
    def ft_eigenvalue_problem_covariance(
            self, cov_t1: np.ndarray, cov_t2: np.ndarray):
        """
            Logic:
            - Solves the generalized eigenvalue problem:
              cov_t1 * w = lambda * cov_t2 * w (Equation 5)
            - Sorts eigenvalues and corresponding eigenvectors in ascending
              order
            - The sorted eigenvectors represent CSP filters ordered by
              discriminative power
            Return:
            - eigvals_sorted (np.ndarray): Sorted eigenvalues in ascending
              order
            - eigvecs_sorted (np.ndarray): Corresponding eigenvectors sorted by
              eigenvalue
        """
        eigvals, eigvecs = scipy.linalg.eig(cov_t1, cov_t2)
        index_vals = np.argsort(eigvals)
        eigvals_sorted = eigvals[index_vals]
        eigvecs_sorted = eigvecs[:, index_vals]
        # self.ft_plot_discriminative_model(eigvals, eigvecs)
        # self.ft_plot_discriminative_model(eigvals_sorted, eigvecs_sorted)
        return eigvals_sorted, eigvecs_sorted

    def ft_plot_discriminative_model(
            self,
            eigvals_sorted: np.ndarray,
            eigvecs_sorted: np.ndarray):
        """
            Logic:
            - Extracts the first and last eigenvectors
              (most discriminative filters)
            - Computes variance of each epoch projected onto these two filters
            - Creates a scatter plot showing class separation
            - Colors epochs: blue for class 2, red for class 3
            Return:
            - None
        """
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

        # Affichage avec des couleurs personnalisées : bleu pour Y==2 et rouge
        # pour Y==3
        colors = ['blue' if y == 2 else 'red' for y in self.Y]
        plt.scatter(var_min, var_max, c=colors, marker='o', alpha=0.7)
        plt.xlabel("Variance sur filtre v_min")
        plt.ylabel("Variance sur filtre v_max")

        plt.show()

    def ft_compute_W_matrix(
            self,
            eigvals_sorted: np.ndarray,
            eigvecs_sorted: np.ndarray):
        """
            Logic:
            - Selects k=n_components//2 smallest and k largest eigenvectors
              (most discriminative)
            - Transposes and stacks them to create the CSP transformation
              matrix W
            - W has shape (2*k, n_channels) = (n_components, n_channels)
            - This matrix projects EEG signals onto CSP space
            Return:
            - W (np.ndarray): CSP transformation matrix with shape
              (n_components, n_channels)
        """
        k = getattr(self, 'n_components', 6) // 2
        eigvecs_sorted = eigvecs_sorted.real
        W_small = eigvecs_sorted[:, :k].T
        W_large = eigvecs_sorted[:, -k:].T
        W = np.vstack([
            W_small,
            W_large
        ])
        return W

    def ft_obtain_csp_signals(self, W: np.ndarray, X: np.ndarray):
        """
            Logic:
            - Projects each epoch X[i] onto CSP space: z_i = W @ X[i]
            - Computes variance along time axis for each CSP component
            - Applies log transform to variance: log(var(z_i))
            - Creates feature vector for each epoch
            Return:
            - features (np.ndarray): CSP feature matrix with shape
              (n_epochs, 2*k)
        """
        features = []
        for i in range(X.shape[0]):
            X_i = X[i]
            z_i = W @ X_i
            f_i = np.log(np.var(z_i, axis=1))
            features.append(f_i)
        features = np.array(features)
        return features

    def ft_return_Y_labels(self):
        """
            Logic:
            - Returns the event labels extracted from the epochs
            - Labels correspond to motor imagery tasks (typically 2 or 3)
            Return:
            - Y (np.ndarray): Array of event labels for each epoch
        """
        return self.Y

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Logic:
            - Stores X and y in self.X and self.Y
            - Reuses ft_compute_covariance_matrices() to compute covariance
            - Reuses ft_eigenvalue_problem_covariance() to solve eigenvalue
              problem
            - Reuses ft_compute_W_matrix() to compute CSP transformation
              matrix W
            Return:
            - self
        """
        self.X = X
        self.Y = y

        cov_t1, cov_t2 = self.ft_compute_covariance_matrices()
        eigvals_sorted, eigvecs_sorted = self.ft_eigenvalue_problem_covariance(
            cov_t1, cov_t2)
        self.W = self.ft_compute_W_matrix(eigvals_sorted, eigvecs_sorted)

        return self

    def transform(self, X: np.ndarray):
        """
            Logic:
            - Reuses ft_obtain_csp_signals() to transform epochs to CSP
              features
            - Projects each epoch onto CSP space and computes log-variance
            Return:
            - features (np.ndarray): CSP feature matrix
        """
        return self.ft_obtain_csp_signals(self.W, X)

    n_components: int
    W: Optional[np.ndarray]
    X: Optional[np.ndarray]
    Y: Optional[np.ndarray]
    epochs: Optional[mne.Epochs]


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

    eigvals_sorted, eigvecs_sorted = csp.ft_eigenvalue_problem_covariance(
        cov_t1, cov_t2)

    W = csp.ft_compute_W_matrix(eigvals_sorted, eigvecs_sorted)

    features = csp.ft_obtain_csp_signals(W, csp.X)
    print(features)


if __name__ == "__main__":
    main()
