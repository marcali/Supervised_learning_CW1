from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from matplotlib import pyplot as plt


DIMENSIONS_TO_18 = np.arange(1, 19, dtype=int)  # k=1...18
TEST_COUNT = 1000
V = 3


######Part 1.1 linear regression######


def polynomial_map(x, k_range) -> np.array:
    k_range = k_range[None, :]
    poly_map = x[:, None] ** (k_range - 1)
    return poly_map


def calculate_weights(X, y) -> np.array:
    pinv = np.linalg.pinv(X.T @ X)  # Find the pseudo inverse
    return pinv @ X.T @ y


def calculate_sse(X, y, w) -> float:
    A = (X @ w - y)
    return A.T @ A


def calculate_mse(X, y_true, w) -> float:
    # The mean square error is the average of the sum of squared errors
    return calculate_sse(X, y_true, w) / X.shape[0]


class LinearRegressionFit(NamedTuple):
    X: np.array
    w: np.array
    sse: float
    mse: float


class PlotData(NamedTuple):
    x_range: np.array
    limits: tuple[float, float]
    ax: plt.Axes


def fit_linear_regression(y, X) -> LinearRegressionFit:
    """
    :param k: degree of polynomial
    :param x: features of dataset
    :param y: tagert value of dataset
    :param feature_map: callable feature map
    :return: (named)tuple of linear regression fit
    """
    w = calculate_weights(X, y)
    sse = calculate_sse(X, y, w)
    mse = sse / X.shape[0]
    return LinearRegressionFit(X=X, w=w, sse=sse, mse=mse)


def fit_linear_regressions(y, X_train_mapped, k_range, plot=None, x_range_mapped=None) -> dict[
    int, LinearRegressionFit]:
    """
    :param y: target value of dataset
    :param X_train_mapped: features of dataset(matrix of shape (n, k)) mapped according to chosen feature map(polynomial or sin)
    :param k_range: range of degrees of polynomial to fit
    :param plot: optional plot data to plot the fit
    :param x_range_mapped: optional x range to plot the fit
    :return: dictionary of linear regression fits
    """
    models = {}
    for k_i, k in enumerate(k_range):
        fit = fit_linear_regression(y, X_train_mapped[:, :k])
        models[k] = fit

        if plot:
            y_prediction = fit.w @ x_range_mapped[:, :k].T
            poly_plot(plot.ax, plot.x_range, y_prediction, fit.mse, plot.limits, k=k)

    return models


def poly_plot(ax, x_range, y_prediction, mse, limits, k):
    ax.plot(x_range, y_prediction, label=f'k = {k}, mse={mse:.2f}')
    ax.set_ylim(limits)
    clean_chart(ax)
    ax.legend(loc='upper center')


def g(x, std_dev=0.07):
    epsilon = np.random.normal(loc=0.0, scale=std_dev, size=x.size)
    return np.sin(2 * np.pi * x) ** 2 + epsilon


def generate_data_points(n_data_points: int) -> tuple[np.array, np.array]:
    x = np.random.uniform(size=n_data_points)
    return x, g(x)


def plot_sin_function_Q2a(S_x, S_y, x_range):
    fig, ax = plt.subplots()

    y_without_noise = g(x_range, std_dev=0)

    ax.plot(x_range, y_without_noise, '-')
    ax.plot(S_x, S_y, '.')
    ax.set_ylim(0, ax.get_ylim()[1])
    clean_chart(ax)
    plt.xlabel("x")
    plt.ylabel("y")
    fig.savefig('SinData_2_a_i.png', dpi=300)


def calculate_test_mse(linear_regression_fits: dict[int, LinearRegressionFit], T_x_mapped: np.array,
                       T_y_actual: np.array, T_x: np.array) -> list[float]:
    """
    Given a dictionary of fitted linear regressions, return an array of the mean square error on the mapped test data

    :param linear_regression_fits: dictionary of linear regression fits
    :param T_x_mapped: mapped test data according to chosen feature map(polynomial or sin)
    :param T_y_actual: target value of test data
    :param T_x: test data
    :return: array of mean square error on the mapped test data
    """
    test_mses = []
    for k, fit in linear_regression_fits.items():
        X = T_x_mapped[:, :k]
        sse = calculate_sse(X=X, y=T_y_actual, w=fit.w)
        mse = sse / T_x.shape[0]
        test_mses.append(mse)
    return test_mses


def run_linear_regressions(map_function: callable, n_runs: int = 100):
    """
    :param map_function: callable feature map(polynomial or sin)
    :param n_runs: number of runs to average over
    :return: test_mses - array of shape (n_runs, k_range), train_mses - array of shape (n_runs, k_range)
    """
    test_mses = []
    train_mses = []
    for run in range(n_runs):
        # Generate new training data
        S_x, S_y = generate_data_points(30)
        S_x_mapped = map_function(S_x, DIMENSIONS_TO_18)

        k18_fitted_polynomials_i = fit_linear_regressions(
            y=S_y, X_train_mapped=S_x_mapped, k_range=DIMENSIONS_TO_18
        )
        train_mses.append([poly.mse for poly in k18_fitted_polynomials_i.values()])

        # Generate new test data
        T_x, T_y_actual = generate_data_points(TEST_COUNT)
        T_x_mapped = map_function(T_x, DIMENSIONS_TO_18)

        test_mses.append(calculate_test_mse(k18_fitted_polynomials_i, T_x_mapped, T_y_actual, T_x))

    return test_mses, train_mses


def sin_map(x: np.array, k: np.array):
    sin_features = np.sin(np.pi * x[:, np.newaxis] * k)
    return sin_features


#####Part 1.3 kernel ridge regression#######


@dataclass
class FoldData:
    fold_index: int
    start_index: int
    stop_index: int
    train_data: np.array
    train_labels: np.array
    test_data: np.array
    test_labels: np.array


def create_folds_data(X_train, Y_train, number_of_folds: int) -> list[FoldData]:
    """
    Create a list of FoldData helper objects which contain all the data needed to define a fold.

    :param X_train: training data
    :param Y_train: training labels
    :param number_of_folds: number of folds to split the data into
    :return: list of FoldData objects
    """
    fold_size = X_train.shape[0] / number_of_folds

    fold_data = []
    for i_fold in range(number_of_folds):
        # Define the fold start/stop as the range of the test data in the fold
        fold_start = int(i_fold * fold_size)
        fold_stop = int((i_fold + 1) * fold_size) if i_fold != number_of_folds - 1 else X_train.shape[0]

        fold_test_data = X_train[fold_start:fold_stop, :]
        fold_test_labels = Y_train[fold_start:fold_stop]

        # The training data is all the non-test data
        fold_train_data = np.vstack((X_train[:fold_start, :], X_train[fold_stop:, :]))
        fold_train_labels = np.concatenate((Y_train[:fold_start], Y_train[fold_stop:]))

        fold_data.append(
            FoldData(
                fold_index=i_fold,
                start_index=fold_start,
                stop_index=fold_stop,
                train_data=fold_train_data,
                train_labels=fold_train_labels,
                test_data=fold_test_data,
                test_labels=fold_test_labels,
            )
        )

    return fold_data


def calc_alpha_star(K, gamma, l, y):
    """
    :param K: The Kernel matrix
    :param gamma: The ridge regression loss function regulariser
    :param l: The number of training examples in the training set
    :param y: y-values of the training set
    :return: The dual regression coefficients
    """
    return np.linalg.pinv(K + gamma * l * np.identity(l)) @ y


def calc_dist_sq(X1, X2):
    """
    Given two matrices with shape (number of points, number of features), calculate a distance-squared matrix
    corresponding to the distances between the points in X1 and the points in X2

    For scalars |x-y|^2 = x^2 + y^2 - 2xy, we are doing the same below but vectorised

    :param X1: np.array of shape (n_points_1, n_features)
    :param X2: np.array of shape (n_points_2, n_features)
    :return: np.array of shape (n_points_1, n_points_2) with each value at [i, j] corresponding to the squared distance
             between the i-th row of X1 and the j-th row of X2
    """
    X1_sq = (X1 ** 2).sum(axis=1, keepdims=True)
    X2_sq = (X2 ** 2).sum(axis=1, keepdims=True)
    X1_X_2 = X1 @ X2.T

    assert X1_sq.shape == (X1.shape[0], 1)
    assert X2_sq.shape == (X2.shape[0], 1)
    assert X1_X_2.shape == (X1.shape[0], X2.shape[0])

    dist = X1_sq + X2_sq.T - 2 * X1_X_2
    return dist


def gaussian_kernel(X1: np.array, X2: np.array, sigma: float) -> np.array:
    """
    Create a gaussian kernel matrix over the points in X1, X2

    :param X1: np.array of shape (n_points_1, n_features)
    :param X2: np.array of shape (n_points_2, n_features)
    :param sigma: float of standard deviation of the gaussian kernel
    :return: np.array of shape (n_points_1, n_points_2) with each value at [i, j] corresponding to the kernel function
             between the i-th row of X1 and the j-th row of X2
    """
    dist = calc_dist_sq(X1, X2)
    K = np.exp(- 0.5 * (dist) / (sigma ** 2))
    assert K.shape == (X1.shape[0], X2.shape[0])

    return K


def dist_squared_vectors(x, y) -> float:
    return (x ** 2 + y ** 2 - 2 * x * y).sum()


def calculate_mse_from_labels(predicted_labels, true_labels) -> float:
    assert predicted_labels.shape == true_labels.shape == (len(predicted_labels),)
    return dist_squared_vectors(predicted_labels, true_labels) / predicted_labels.shape[0]


def get_best_gamma_sigma_from_cross_validation_error(mse_cv_test: np.array, gammas, sigmas) -> tuple[float, float, int, int]:
    mse_over_folds = mse_cv_test[:, :, :].mean(axis=0)  # average over the folds
    index_min_error_flat = np.argmin(mse_over_folds)  # get the index of the minimum cross validation error
    # convert into the 2-d index of gamma and sigma
    best_gamma_index, best_sigma_index = np.unravel_index(
        index_min_error_flat, mse_over_folds.shape
    )
    best_gamma, best_sigma = gammas[best_gamma_index], sigmas[best_sigma_index]
    return best_gamma, best_sigma, best_gamma_index, best_sigma_index


def clean_chart(ax: plt.Axes):
    ax.minorticks_on()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


#####part 1.3 k nearest neighbour#####


def sample_h() -> tuple[np.array, np.array]:
    """Sampling 100 centers uniformly at random from [0, 1]2 with 100 corresponding labels sampled uniformly at
    random from {0, 1}"""
    centres = np.random.uniform(size=(100, 2))
    labels = np.random.choice([0, 1], size=(100,))
    return centres, labels


def find_nearest_neighbours(X_test, X_train, Y_train, k) -> np.array:
    """
    Find the nearest neighbours in X_train for each point in X_test by getting the class of the neighbours sorted by
    distance then getting the classification given by the first k labels

    :param X_test: np.array of test points
    :param X_train: np.array of train points
    :param Y_train: np.array of train labels
    :param k: number of nearest neighbours to consider
    :return: np.array of classification of test points
    """
    labels_by_distance = get_neighbour_class_by_distance(X_test, X_train, Y_train)
    return get_knn_classification(labels_by_distance, k)


def get_knn_classification(labels_by_distance: np.array, k: int):
    """
    :param labels_by_distance: 2d array of binary int values (e.g. 0 or 1)
        where the element [i, j] of the array is the class of the j-th nearest neighbour of the i-th test point
    :param k: the number of nearest neighbours we're interested in
    :return: 1d array of binary int values corresponding to the classification given by the k nearest neighbours
    """
    closest_k_labels = labels_by_distance[:, :k]

    # corner case - assign a random label if there is a tie by adding small (<1/(k+1) so it only affects the tie)
    # random noise
    small_random_noise = (np.random.random(size=closest_k_labels.shape[0]) - 0.5) / (k + 1)
    avg_vote = closest_k_labels.mean(axis=1) + small_random_noise

    return np.round(avg_vote)


def get_neighbour_class_by_distance(X_test, X_train, Y_train):
    """
    Given an array of test points find rank the training points by their distance to each test point
    and return an array of the training class corresponding to each of those test points
    :param X_test:
    :param X_train:
    :param Y_train:
    :return: array of int (e.g. 0 or 1)
        where the element [i, j] of the array is the class of the j-th nearest neighbour of the i-th test point
    """
    dist = calc_dist_sq(X1=X_test, X2=X_train)

    points_by_distance = np.argsort(dist, axis=1)  # get the indices of the points sorted by distance
    assert points_by_distance.shape == (X_test.shape[0], X_train.shape[0])

    return Y_train[points_by_distance] # get the classes corresponding to the indices of the points sorted by distance


def generate_data_from_p_h(n_points, p_h_centres, p_h_labels, k):
    """
    :param n_points: number of points to generate
    :param p_h_centres: centres of the distribution, x1&x2 for each point
    :param p_h_labels: labels of the distribution, y for each point {0, 1}
    :param k: number of nearest neighbours to use
    :return: x, y - x is the points, y is the labels
    """
    x = np.random.uniform(size=(n_points, 2))
    y = np.zeros(shape=(n_points,))
    coin_flips = np.random.choice([0, 1], size=n_points, p=[0.2, 0.8])
    heads = coin_flips == 1
    tails = coin_flips == 0

    y[tails] = np.random.choice([0, 1], size=n_points)[tails]
    y[heads] = find_nearest_neighbours(X_test=x, X_train=p_h_centres, Y_train=p_h_labels, k=k)[heads]
    return x, y


def run_nearest_neighbours(n_training_points: np.array, n_test_points, n_runs, n_ks) -> np.array:
    """
    For each k∈{1,...,49} Do 100 runs ...
    Sample a h from pH
        - this is a repeat of part 1 to get 100 points
    Build a k-NN model with 4000 training points sampled from ph(x,y)
        - this is our new thing with biased coin where 80% of the time we take from the h we just got, 20% of the time we flip randomly
    Run k-NN estimate generalisation error (for this run) using 1000 test points sampled from ph(x,y)
        - we just see how from our prediction equal from our ytest and average to get a rate e.g. (y_test == y_pred).mean()
    The estimated generalisation error (y-axis) is then the mean of these 100 ‘‘run’’ generalisation errors.

    Note - have done as 100 runs and for each run we're doing 50 different ks rather than the other way around.
    The reason is that we can use the same dataset for all 50 k values which makes it much faster but for any given k
    value we've still looked at the shape across 100 different datasets.

    :param n_training_points: array of number of training points to use
        e.g. n_training_points = [4000] for protocol A
            n_training_points = [100, 500, 1000, ..., 4000] for protocol B
    :param n_test_points: number of test points to use
    :param n_runs: number of runs to average over
    :param n_ks: number of k values to test
    :return: error_rates - array of shape (n_training_points, n_runs, n_ks)
    """
    ks = np.arange(1, n_ks + 1, step=1)
    error_rates = np.zeros(shape=(n_training_points.shape[0], n_runs, n_ks))

    for i_run in range(n_runs):
        h_x, h_y = sample_h()

        # Generate all the data points for the largest m so we can just slice in to get the data for the smaller m
        max_n_training_points = n_training_points[-1]
        X_train, Y_train = generate_data_from_p_h(n_points=max_n_training_points, p_h_centres=h_x, p_h_labels=h_y, k=V)
        X_test, Y_test_true = generate_data_from_p_h(n_points=n_test_points, p_h_centres=h_x, p_h_labels=h_y, k=V)

        for i_m, m in enumerate(n_training_points):

            # We slice in to only the training points we need for this particular m
            X_train_m, Y_train_m = X_train[:m, :], Y_train[:m]
            X_test_m, Y_test_true_m = X_test[:m, :], Y_test_true[:m]

            labels_by_distance = get_neighbour_class_by_distance(X_test_m, X_train_m, Y_train_m)

            for i_k, k in enumerate(ks):
                Y_test_prediction = get_knn_classification(labels_by_distance, k)
                # Get the error rate as the mean of the number of times that the prediction is wrong
                error_rates[i_m, i_run, i_k] = (Y_test_prediction != Y_test_true_m).mean()
    return error_rates


def plot_and_save_mse(x_range, y_errors, save_path=None):
    fig, ax = plt.subplots()
    ax.plot(x_range, y_errors, '-')
    ax.set_xlabel('k')
    ax.set_ylabel('ln(average mean square error)')
    ax.set_xlim(0, 18)
    if save_path:
        fig.savefig(save_path, dpi=300)
