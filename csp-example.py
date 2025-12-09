import numpy as np
import random
import scipy

def ft_compute_covariance_matrices(X, Y):
        cov_pos = []
        cov_neg = []

        for i in range(X.shape[0]):
            if Y[i] == 2:
                result_positive = X[i] @ X[i].T

                print(result_positive.shape)
                # flat signal covariance matrix
                if np.trace(result_positive) < 1e-12:
                    continue
                result_pos_norm = result_positive / np.trace(result_positive)
                print(result_pos_norm.shape)
                cov_pos.append(result_pos_norm)
                

            elif Y[i] == 3:
                result_negative = X[i] @ X[i].T
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


def ft_eigenvalue_problem_covariance(cov_t1: np.ndarray, cov_t2: np.ndarray):
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
    return eigvals_sorted, eigvecs_sorted


def ft_compute_W_matrix(eigvals_sorted: np.ndarray, eigvecs_sorted: np.ndarray):
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
    k = 6 // 2
    eigvecs_sorted = eigvecs_sorted.real
    W_small = eigvecs_sorted[:, :k].T
    W_large = eigvecs_sorted[:, -k:].T
    W = np.vstack([
        W_small,
        W_large
    ])
    return W

def ft_obtain_csp_signals(W: np.ndarray, X: np.ndarray):
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



def main():
    n_epochs = 20
    n_channels = 64
    n_times = 100

    example_array = np.random.rand(n_epochs, n_channels, n_times)

    labels = [random.choice([2, 3]) for _ in range(n_epochs)]
    example_labels = np.asarray(labels, dtype=np.int32)

    print("*************** Initial matrix **************")
    print(example_array)
    print("Shape X:", example_array.shape)

    print("*************** Initial labels **************")
    print(example_labels)
    print("Shape y:", example_labels.shape)

    cov_pos, cov_neg = ft_compute_covariance_matrices(example_array, example_labels)

    print("*************** Σ+ (T1) **************")
    print(cov_pos)
    print("Shape Σ+ :", cov_pos.shape)

    print("*************** Σ- (T2) **************")
    print(cov_neg)
    print("Shape Σ- :", cov_neg.shape)


    eigvals_sorted, eigvecs_sorted = ft_eigenvalue_problem_covariance(cov_pos, cov_neg)

    print("*************** Eigen values **************")
    print(eigvals_sorted)
    print(eigvals_sorted.shape)
    print("*************** Eigen vectors **************")
    print(eigvecs_sorted)
    print(eigvecs_sorted.shape)

    W = ft_compute_W_matrix(eigvals_sorted, eigvecs_sorted)

    print("*************** W **************")
    print(W)
    print(W.shape)

    features = ft_obtain_csp_signals(W, example_array)
    
    print("*************** X CSP **************")
    print(features)
    print(features.shape)


if __name__ == "__main__":
    main()
