import numpy as np
import pandas as pd

np.int = int  # Fix deprecated

import pickle
from joblib import load
import dill

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import _tree, DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.stats import kruskal, kstest, shapiro, skewnorm
from scipy.spatial import distance

import torch
from torch import nn

from typing import Iterator, Optional
import random

from collections import Counter

import shap

from typing import Callable, List, Tuple, Union

from lime.lime_tabular import LimeTabularExplainer
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

import copy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc

## AUXILIARY ###
DATA_DIRECTORY = '/serialised'


def save_to_pickle(obj, filename, directory=DATA_DIRECTORY):
    """
    Save an object to a pickle file.

    Parameters:
    - obj: Object to save
    - filename: Name of the pickle file
    - directory: Directory to save the pickle file (default is DATA_DIRECTORY)

    Returns:
    - None
    """
    filepath = os.path.join(directory, filename)

    # Check if the object is a pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        obj.to_pickle(filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

    print(f"Saved to {filepath}")


def load_from_pickle(filename, directory=DATA_DIRECTORY):
    """
    Load an object from a pickle file using pandas, pickle, or joblib as a fallback.

    Parameters:
    - filename: Name of the pickle file
    - directory: Directory where the pickle file is located (default is DATA_DIRECTORY)

    Returns:
    - Object loaded from the pickle file
    """
    filepath = os.path.join(directory, filename)

    try:
        import pandas as pd
        obj = pd.read_pickle(filepath)
    except Exception as e:  # Catch any exception from pandas
        print(f"Failed to load pickle with pandas due to: {e}")
        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
        except Exception as e:  # Catch any exception from standard pickle
            print(f"Failed to load pickle with standard pickle due to: {e}")
            try:
                obj = load(filepath)  # Try with joblib
            except Exception as e:
                print(f"Failed to load pickle with joblib due to: {e}")
                try:
                    obj = dill.load(filepath)  # Try with joblib
                except Exception as e:
                    print(f"Failed to load pickle with dill due to: {e}")

                    raise e  # Re-raise the exception if all methods fail

    return obj


def get_stats(df: pd.DataFrame, by: str):
    grouped = df.groupby(by)
    summary_stats = grouped.agg(['count', 'mean', 'std', 'min', 'max'])
    return summary_stats


## MODEL ###

class IrisAutoencoder3(nn.Module):
    def __init__(self):
        super(IrisAutoencoder3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 8),  # Increase complexity by adding more neurons
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(8, 4),
            nn.ReLU(True),
            nn.Linear(4, 2),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(True),
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(8, 4),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def print_loss(X_tensor: torch.Tensor, model: nn.Module, label: str = '') -> None:
    _criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        loss = _criterion(output, X_tensor)

    print(f"[{label}] test Loss= {loss.item()}")


def get_losses(X_tensor: torch.Tensor, model: nn.Module, label: str = '') -> Tuple[List[float], float]:
    losses = []
    _criterion = nn.MSELoss()
    with torch.no_grad():
        for sample in X_tensor:
            output = model(sample)
            loss = _criterion(output, sample)
            losses.append(loss.item())

    # Choose a cutoff value
    cutoff = sum(losses) / len(losses)
    if label:
        print("[", label, '] Cutoff Loss:', cutoff)
    else:
        print('Cutoff Loss:', cutoff)

    return losses, cutoff


## CLASSIFIER ###

def find_best_threshold(losses_normal: List[float], losses_anomaly_1: List[float], normal_label: str,
                        anomaly_label: str, do_plot: bool = True) -> float:
    # Create histograms for both distributions
    hist_normal, bins_normal = np.histogram(losses_normal, bins=100, density=True)
    hist_anomaly_1, bins_anomaly_1 = np.histogram(losses_anomaly_1, bins=100, density=True)

    best_score = -np.inf
    best_threshold = None

    # Compute quality measures for each threshold
    for i in range(1, len(bins_normal)):
        threshold = bins_normal[i]
        score = sum(hist_normal[j] for j in range(i)) + sum(hist_anomaly_1[j] for j in range(i, len(hist_anomaly_1)))

        if score > best_score:
            best_score = score
            best_threshold = threshold

    # Visualization using the provided plot code
    if do_plot:
        plt.figure(figsize=(12, 8))
        plt.hist(losses_normal, bins=30, label=normal_label)
        plt.hist(losses_anomaly_1, bins=30, label=anomaly_label)
        plt.axvline(best_threshold, color='r', linestyle='dashed', linewidth=2,
                    label=f'Best Threshold: {best_threshold:.2f}')
        plt.title('Distribution of Losses')
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.legend()  # Display the labels
        plt.show()

    return best_threshold


def assess_gaussianity_and_percentile(data: List[float], percentile: float = 0.75) -> Tuple[float, str]:
    """
    Assesses the gaussianity of the provided data and returns the value of the desired percentile.

    Parameters:
    - data: List of data points.
    - percentile: Desired percentile to compute. Defaults to 0.75.

    Returns:
    - Tuple of (value at desired percentile, assessment of gaussianity).
    """

    # Compute the desired percentile
    percentile_value = np.percentile(data, percentile * 100)

    # Perform the Shapiro-Wilk test for normality
    stat, p = shapiro(data)

    if p > 0.05:
        assessment = "Data can be considered Gaussian (fail to reject H0)."
    else:
        assessment = "Data does not look Gaussian (reject H0)."

    return percentile_value, assessment


## TREE ###

def unscale_data(scaled_data: Union[np.ndarray, torch.Tensor, pd.DataFrame], a_scaler: StandardScaler) -> pd.DataFrame:
    """Unscale data that has been previously scaled with a StandardScaler.

    Args:
        scaled_data (Union[np.ndarray, torch.Tensor, pd.DataFrame]): The scaled data. Can be a numpy array, Torch tensor or a pandas DataFrame.
        a_scaler (StandardScaler): The StandardScaler instance that was used to scale the data.

    Returns:
        pd.DataFrame: The unscaled data as a pandas DataFrame.
    """
    # If the data is a DataFrame, convert it to a numpy array
    if isinstance(scaled_data, pd.DataFrame):
        print("Got pd.DataFrame")
        scaled_data = scaled_data.to_numpy()
    # If the data is a Torch tensor, convert it to a numpy array
    elif isinstance(scaled_data, torch.Tensor):
        print("Got torch.Tensor")
        scaled_data = scaled_data.numpy()
    else:
        print(f"Got {type(scaled_data)}")

    # Use the inverse_transform method to unscale the data
    unscaled_data = a_scaler.inverse_transform(scaled_data)

    # Convert the unscaled data to a DataFrame
    unscaled_df = pd.DataFrame(unscaled_data, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                                       'petal width (cm)'])
    print(unscaled_df.info())

    return unscaled_df


def plot_decision_areas(df: pd.DataFrame, dimmension_1: str, dimmension_2: str, a_classifier) -> None:
    """
    Plot the decision areas for a given DecisionTreeClassifier on a scatter plot of two specified dimensions.

    Parameters:
    - df: DataFrame containing the data.
    - dimmension_1: First dimension (or feature) for the x-axis.
    - dimmension_2: Second dimension (or feature) for the y-axis.
    - classifier: The DecisionTreeClassifier object.
    """

    # Extract unique target values
    unique_targets = df['target'].unique()

    # Create a scatter plot for the given dimensions
    for target_value in unique_targets:
        subset = df[df['target'] == target_value]
        plt.scatter(subset[dimmension_1], subset[dimmension_2], label=f'class: {target_value}')

    # Extract the tree's decision thresholds
    thresholds = a_classifier.tree_.threshold
    features = a_classifier.tree_.feature
    feature_names = list(df.columns[:-1])  # Assuming last column is 'target'

    # Plot decision areas based on the decision thresholds of the classifier
    for feature, threshold in zip(features, thresholds):
        if feature_names[feature] == dimmension_1:
            plt.axvspan(df[dimmension_1].min(), threshold, facecolor='red', alpha=0.2)
        elif feature_names[feature] == dimmension_2:
            plt.axhspan(df[dimmension_2].min(), threshold, facecolor='red', alpha=0.2)

    plt.xlabel(dimmension_1)
    plt.ylabel(dimmension_2)
    plt.legend()
    plt.title(f'{dimmension_1} vs. {dimmension_2}')
    plt.show()


## SIMULATED TS ###

def simulate_time_series(N1: int, N2: int, N3: int, N123: int, N4: int, p: float,
                         X_normal: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                         X_anomaly_1: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                         X_anomaly_2: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                         seed: Optional[int] = None) -> Iterator:
    """
    Simulate time series with changing distribution over time. The distribution changes in five phases:
    - During the first N1 samples, only normal samples are returned.
    - During the next N2 samples, the ratio of anomaly 1 samples gradually increases until it reaches 1.
    - During the next N123 samples, all three classes are present.
    - During the next N3 samples, the ratio of anomaly 2 samples gradually increases until it reaches 1.
    - During the last N4 samples, only anomaly 2 samples are returned.

    In each phase, the sample type is chosen randomly, with the probability of each type changing over time as described.

    [... rest of the documentation ...]
    """
    initial_seed = random.getstate()
    if seed is not None:
        random.seed(seed)

    i = 0
    j = 0
    k = 0
    l = 0
    m = 0

    # Generate N1 normal samples
    for i in range(N1):
        yield random.choice(X_normal), 'N', i + j + k + l + m

    # Generate N2 mixed normal and anomaly 1 samples
    for j in range(N2):
        anomaly_1_ratio = -p + j / N2
        if random.random() < anomaly_1_ratio:
            yield random.choice(X_anomaly_1), 'A1', i + j + k + l + m
        else:
            yield random.choice(X_normal), 'N', i + j + k + l + m

    # Generate N123 samples from all three classes
    for k in range(N123):
        norm_ratio = p - k / N123
        if random.random() < norm_ratio:
            yield random.choice(X_normal), 'N', i + j + k + l + m
        else:
            _cl = random.choice(['A1', 'A2'])
            if _cl == 'A1':
                yield random.choice(X_anomaly_1), 'A1', i + j + k + l + m
            else:
                yield random.choice(X_anomaly_2), 'A2', i + j + k + l + m

    # Generate N3 mixed anomaly 1 and anomaly 2 samples
    for l in range(N3):
        anomaly_2_ratio = l / N3
        if random.random() < anomaly_2_ratio:
            yield random.choice(X_anomaly_2), 'A2', i + j + k + l + m
        else:
            yield random.choice(X_anomaly_1), 'A1', i + j + k + l + m

    # Generate N4 anomaly 2 samples
    for m in range(N4):
        yield random.choice(X_anomaly_2), 'A2', i + j + k + l + m

    random.setstate(initial_seed)


def structured_list_to_dataframe(
        a_time_series: List[Tuple[Union[int, float], Union[List[float], np.ndarray]]]) -> pd.DataFrame:
    """
    Convert a structured list into a pandas DataFrame.

    Parameters:
    - a_time_series: List with structure containing iris features, sample type, and index.

    Returns:
    - DataFrame representation of the structured list.
    """

    # Extracting column names for iris dataset
    iris_features = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']

    # Preparing the list of dictionaries
    data_list = []
    for record in a_time_series:
        data_dict = {}
        for idx, feature in enumerate(iris_features):
            data_dict[feature] = record[0][idx]
        data_dict['sample_type'] = record[1]
        data_dict['index'] = record[2]
        data_list.append(data_dict)

    # Converting list of dictionaries to DataFrame
    df = pd.DataFrame(data_list)

    return df


## CLASSIFIER ON AUTOENCODER ###

class AnomalyClassifier(BaseEstimator, ClassifierMixin):
    """
    Takes into account Right-skew distribution ("Rozkład prawoskośny")
    """

    def __init__(self,
                 autoencoder_model: torch.nn.Module,
                 get_losses: Callable[[Union[np.ndarray, torch.Tensor, pd.DataFrame], torch.nn.Module], List[float]],
                 loss_threshold: float,
                 a_scaler) -> None:
        self.autoencoder_model = autoencoder_model
        self.get_losses = get_losses
        self.loss_threshold = loss_threshold
        self.scaler = a_scaler
        self.losses_normal = None
        self.skew_params = None
        self.ks_statistic = None
        self.ks_p_value = None
        self.cdf_threshold = None

    def fit(self, X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
            y: Optional[np.ndarray] = None) -> "AnomalyClassifier":
        # Currently, the fit method doesn't do anything.
        return self

    def _data_to_tensor(self, X: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> torch.Tensor:
        # If X is a DataFrame, convert it to numpy array and scale
        if isinstance(X, pd.DataFrame):
            X = X.values
            # Standardize the input
            X = self.scaler.transform(X)

        # Convert X to Tensor if it's a numpy array
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        return X

    def initialise_predict_proba(self, X: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> "AnomalyClassifier":
        X = self._data_to_tensor(X)

        # Calculate losses
        self.losses_normal, _ = self.get_losses(X, self.autoencoder_model)
        print(
            f"losses min= {np.min(self.losses_normal)}, max= {np.max(self.losses_normal)}, threshold= {self.loss_threshold}")

        # Fitting a skewnorm distribution to our data
        self.skew_params = skewnorm.fit(self.losses_normal)
        # Using kstest to check the goodness of fit
        self.ks_statistic, self.ks_p_value = kstest(self.losses_normal, 'skewnorm', self.skew_params)
        print(f"KS Test Result: Statistic = {self.ks_statistic}, P-Value = {self.ks_p_value}")
        if self.ks_p_value < 0.05:
            print(f"Data does NOT seem to fit a skew-normal distribution (KS Test P-value = {self.ks_p_value:.5f}).")
        else:
            print(f"Data seems to fit a skew-normal distribution (KS Test P-value = {self.ks_p_value:.5f}).")

        # Calculate CDF for threshold value and store in the class attribute
        self.cdf_threshold = skewnorm.cdf(self.loss_threshold, *self.skew_params)

        # Create the histogram of the data
        plt.figure(figsize=(12, 8))
        plt.hist(self.losses_normal, bins=50, density=True, alpha=0.6, color='b', label='Observed Losses')
        # Plot the PDF of the skew-normal distribution we've fitted
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = skewnorm.pdf(x, *self.skew_params)
        plt.plot(x, p, 'k', linewidth=2, label='Fitted Skew-Normal PDF')
        # Plot the CDF of the skew-normal distribution we've fitted
        cdf = skewnorm.cdf(x, *self.skew_params)
        # Add a green dashed line for the CDF
        plt.plot(x, cdf, 'g--', linewidth=2, label='Fitted Skew-Normal CDF')
        # Add a vertical line at the loss_threshold
        plt.axvline(self.loss_threshold, color='magenta', linestyle='dashed', linewidth=2, label='Loss Threshold')
        title = f"Fit results: skew = {self.skew_params[0]:.4f}, loc = {self.skew_params[1]:.4f}, scale = {self.skew_params[2]:.4f}"
        plt.title(title)
        plt.legend()
        plt.show()

        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> np.ndarray:
        X = self._data_to_tensor(X)

        losses, _ = self.get_losses(X, self.autoencoder_model)
        return np.array([0.0 if l < self.loss_threshold else 1.0 for l in losses])

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> np.ndarray:
        X = self._data_to_tensor(X)

        losses, _ = self.get_losses(X, self.autoencoder_model)
        return self._loss_to_proba(losses)

    def _loss_to_proba(self, losses: List[float]) -> np.ndarray:
        """
        Convert losses to probabilities using the knowledge about the normal class distribution.
        """
        cdf_values = skewnorm.cdf(losses, *self.skew_params)

        probas = []
        for cdf_val, loss in zip(cdf_values, losses):
            if loss < self.loss_threshold:
                prob_norm = 0.5 + 0.5 * (self.cdf_threshold - cdf_val) / self.cdf_threshold
                prob_anomal = 1 - prob_norm
            else:
                prob_anomal = 0.5 + 0.5 * (cdf_val - self.cdf_threshold) / (1.0 - self.cdf_threshold)
                prob_norm = 1 - prob_anomal

            probas.append([prob_norm, prob_anomal])

        return np.array(probas)


## FEATURE IMPORTANCE'S: SHAP, LIME ###
def compute_feature_importance_v2_start(a_time_series_df, a_model, a_scaler, explainer_type="SHAP", window=100,
                                        slide=25):
    """
    Compute global feature importance values using a sliding window.

    Parameters:
    - time_series_df: The time series dataframe
    - model: Trained model for which we are computing the feature importance
    - a_scaler: scaler
    - explainer_type: Type of explainer to use ("SHAP" or "LIME" etc.)
    - window: Window size
    - slide: Slide size

    Returns:
    - DataFrame containing start, end, iteration, and feature importance values
    """

    feature_names = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']

    background_data = a_time_series_df[feature_names].iloc[:100]  # assumption: we use only first window
    background_data = a_scaler.transform(background_data)

    results = []

    print(f"explainer_type= {explainer_type}")
    if explainer_type == "SHAP":
        explainer = shap.KernelExplainer(a_model.predict, background_data, silent=True)
    elif explainer_type == "LIME":
        explainer = LimeTabularExplainer(background_data, feature_names=feature_names, class_names=[0, 1],
                                         mode='classification')  # Assuming binary classification
    else:
        raise ValueError(f"Unsupported explainer type: {explainer_type}")

    for start in range(0, len(a_time_series_df) - window + 1, slide):
        end = start + window

        data_window = a_time_series_df.iloc[start:end]
        X_window = data_window[['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']]
        X_window = a_scaler.transform(X_window)

        if explainer_type == "SHAP":
            shap_values = explainer.shap_values(X_window, silent=True)
            mean_importances = np.abs(shap_values).mean(axis=0)

        elif explainer_type == "LIME":
            explanations = [explainer.explain_instance(instance, a_model.predict_proba, num_features=len(feature_names))
                            for instance in X_window]
            lime_importances = np.array(
                [[exp.local_exp[1][i][1] for i in range(len(feature_names))] for exp in explanations])
            mean_importances = np.abs(lime_importances).mean(axis=0)

        # add more explainers if needed

        results.append([start, end] + list(mean_importances))

    columns = ['start', 'end', 'sepal_length_cm_importance', 'sepal_width_cm_importance', 'petal_length_cm_importance',
               'petal_width_cm_importance']
    return pd.DataFrame(results, columns=columns)


def plot_importance(df_importance):
    plt.figure(figsize=(15, 10))

    for column in df_importance.columns[2:]:
        plt.plot(df_importance['start'], df_importance[column], label=column)

    plt.title('Feature Importances over Time')
    plt.xlabel('Time')
    plt.ylabel('Importance')
    plt.legend()
    plt.show()


## CHANGE POINTS IN TS EXPLANATIONS ###
def test_feature_importance_difference(df: pd.DataFrame, breakpoint: int, significance_level: float) -> Tuple[
    float, str]:
    """
    Test if there are significant differences in the feature importance values across predefined segments using the Kruskal-Wallis test.

    Parameters:
    - df: DataFrame containing feature importance data.
    - breakpoint: The point to split the data into segments.
    - significance_level: The level of significance for the test.

    Returns:
    - Tuple containing the p-value and the interpretation message.
    """

    # Split the data into predefined segments based on the breakpoint
    groups = [
        df.loc[df['start'] < breakpoint, 'sepal_length_cm_importance'].values,
        df.loc[(df['start'] >= breakpoint) & (df['start'] < 2 * breakpoint), 'sepal_length_cm_importance'].values,
        df.loc[(df['start'] >= 2 * breakpoint) & (df['start'] < 2.5 * breakpoint), 'sepal_length_cm_importance'].values,
        df.loc[df['start'] >= 2.5 * breakpoint, 'sepal_length_cm_importance'].values
    ]

    h_stat, p_val = kruskal(*groups)

    if p_val < significance_level:
        message = "There are statistically significant differences in the feature importance values across the segments."
    else:
        message = "No significant difference in the feature importance values across the segments."

    return p_val, message


def detect_change_points_isolation_forest(df_importance: pd.DataFrame, contamination: float,
                                          features_to_plot: List[str]) -> List[int]:
    """
    Detect change points in feature importance using Isolation Forest and visualize the results.

    Parameters:
    - df_importance: DataFrame containing feature importance for each window.
    - contamination: Proportion of outliers in the data set.
    - features_to_plot: List of feature names to be plotted.

    Returns:
    - None
    """

    # Prepare the data
    X = df_importance.values

    # Train the Isolation Forest model
    iso_forest = IsolationForest(
        contamination=contamination)  # contamination parameter defines the proportion of outliers
    anomaly_scores = iso_forest.fit_predict(X)

    # Change points are where anomaly_scores are -1 (outliers)
    change_points_iso_forest = np.where(anomaly_scores == -1)[0]

    print("Detected change points at:", change_points_iso_forest)

    if features_to_plot:
        # Visualization
        plt.figure(figsize=(20, 12))

        # Plot feature importance for selected features
        for feature in features_to_plot:
            plt.plot(df_importance[feature], label=f'Feature Importance for {feature}')

        # Mark change points
        for feature in features_to_plot:
            plt.scatter(change_points_iso_forest, df_importance[feature].iloc[change_points_iso_forest], marker='o',
                        label=f'Change Points for {feature}')
            plt.vlines(change_points_iso_forest, ymin=df_importance[feature].min(), ymax=df_importance[feature].max(),
                       colors='red', linestyles='dashed')

        plt.legend()
        plt.title('Feature Importance with Detected Change Points')
        plt.show()

    return change_points_iso_forest.tolist()


def detect_change_points_cumsum(data, threshold=1):
    """
    Detect change points using the CUMSUM method.

    Parameters:
    - data: Series or list of data points
    - threshold: Threshold value to detect change

    Returns:
    - List of indices where change points were detected
    """

    mean_val = np.mean(data)
    cusum = np.cumsum(data - mean_val)

    # Detect change points where the CUSUM value exceeds the threshold
    change_points = np.where(np.abs(cusum) > threshold)[0]

    return change_points


## INNER STABILITY ###

def compute_stability_in_windows_v5_choose_background(
        a_time_series_df: pd.DataFrame,
        a_df_importance_shap: pd.DataFrame,
        a_model: any,
        a_scaler: any,
        n_comparisons: int = 100,
        explainer_type: str = "SHAP",
        background_data_source: str = 'start') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute stability of explanations in windows.

    Parameters:
    - a_time_series_df: The time series dataframe
    - a_df_importance_shap: DataFrame of computed importance values; only indexes are used
    - a_model: The trained model
    - a_scaler: The data scaler
    - n_comparisons: Number of pairs to compare for stability calculation
    - explainer_type: The type of explainer to use ("SHAP" or "LIME")
    - background_data_source: Source of the background data ('start' or 'current_window')

    Returns:
    - DataFrame of stability measures for each window
    - List of importance values for individual observations
    """
    if background_data_source in ('start', 'current_window', 'previous_window'):
        print(f"background_data_source= {background_data_source}")
    else:
        raise ValueError(
            f"Unsupported background data source type provided: {background_data_source}. Please choose 'start', 'current_window' or 'previous_window'.")

    stability_results = []
    individual_importance_values = []

    feature_names = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']

    background_data = a_time_series_df[feature_names].iloc[:100]  # assumption: we use only first window
    background_data = a_scaler.transform(background_data)

    print(f"explainer_type= {explainer_type}")
    if explainer_type == "SHAP":
        explainer = shap.KernelExplainer(a_model.predict, background_data, silent=True)
    elif explainer_type == "LIME":
        explainer = LimeTabularExplainer(background_data, feature_names=feature_names, class_names=[0, 1],
                                         mode='classification')
    else:
        raise ValueError(f"Unsupported explainer type provided: {explainer_type}. Please choose 'SHAP' or 'LIME'.")

    previous_window = None  # Initialize a variable to store the previous window
    for idx, row in a_df_importance_shap.iterrows():
        start, end = int(row['start']), int(row['end'])

        # Print the size of the first window
        if idx == 0:
            print(f"Size of the first window: {end - start}")

        data_window = a_time_series_df.iloc[start:end]
        X_window = data_window[feature_names]
        X_window_scaled = a_scaler.transform(X_window)

        # Determine the background data based on the source
        if background_data_source == 'current_window':
            background_data = a_scaler.transform(X_window)
        elif background_data_source == 'previous_window':
            # Use the previous window if available, otherwise use the current window
            background_data = previous_window if previous_window is not None else a_scaler.transform(X_window)
        # else:
        #     raise ValueError("Invalid background_data_source")

        # Update the previous window for the next iteration
        previous_window = a_scaler.transform(X_window)

        # Initialize the explainer based on the explainer type
        if explainer_type == "SHAP":
            explainer = shap.KernelExplainer(a_model.predict, background_data, silent=True)
        elif explainer_type == "LIME":
            explainer = LimeTabularExplainer(background_data, feature_names=feature_names, class_names=[0, 1],
                                             mode='classification')

        importance_values_for_window = []

        if explainer_type == "SHAP":
            for i in range(X_window_scaled.shape[0]):
                importance_i = explainer.shap_values(X_window_scaled[i].reshape(1, -1), silent=True)
                importance_values_for_window.append(importance_i[0])
                individual_importance_values.append((start, end, start + i) + tuple(importance_i[0]))

        elif explainer_type == "LIME":
            explanations = [explainer.explain_instance(instance, a_model.predict_proba, num_features=len(feature_names))
                            for instance in X_window_scaled]
            lime_importances = np.array(
                [[exp.local_exp[1][i][1] for i in range(len(feature_names))] for exp in explanations])
            importance_values_for_window.extend(lime_importances)
            for i, importance_values in enumerate(importance_values_for_window):
                individual_importance_values.append((start, end, start + i) + tuple(importance_values))
        # add more explainers

        observations_distance_sum = 0
        explanations_distance_sum = 0

        for _ in range(n_comparisons):
            i, j = np.random.choice(range(X_window_scaled.shape[0]), 2, replace=False)

            observation_distance = distance.euclidean(X_window_scaled[i], X_window_scaled[j])
            explanation_distance = distance.euclidean(importance_values_for_window[i], importance_values_for_window[j])

            observations_distance_sum += observation_distance
            explanations_distance_sum += explanation_distance

        stability_for_window = explanations_distance_sum / observations_distance_sum
        stability_results.append(
            (start, end, observations_distance_sum, explanations_distance_sum, stability_for_window))

    stability_columns = ['start', 'end', 'observations_distance_sum', 'explanations_distance_sum',
                         'stability_for_a_window']
    importance_columns = ['start', 'end', 'observation_number', 'sepal_length_cm_importance',
                          'sepal_width_cm_importance', 'petal_length_cm_importance', 'petal_width_cm_importance']

    return pd.DataFrame(stability_results, columns=stability_columns), pd.DataFrame(individual_importance_values,
                                                                                    columns=importance_columns)


def plot_stability(stability_df):
    # Tworzenie nowej figury i osi
    plt.figure(figsize=(12, 6))

    # Definiowanie danych
    x_axis = range(len(stability_df))
    y_axis = stability_df['stability_for_a_window']

    # Tworzenie wykresu
    plt.plot(x_axis, y_axis, marker='o', linestyle='-', color='b')

    # Dodawanie etykiet osi, tytułu i legenda
    plt.xlabel('Window Number')
    plt.ylabel('Stability')
    plt.title('Stability Over Time Windows')
    plt.legend(['Stability for a Window'])

    # Wyświetlanie wykresu
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def get_decision_tree_similarities(tree1, tree2, X_background):
    _feature_names = list(X_background.columns)

    tree_sim__feature_importance = decision_tree_similarity(tree1, tree2, X_background=None,
                                                            feature_names=_feature_names, method='feature_importance')
    tree_sim__predictions = decision_tree_similarity(tree1, tree2, X_background=X_background,
                                                     feature_names=_feature_names, method='predictions')
    tree_sim__structure = decision_tree_similarity(tree1, tree2, X_background=None, feature_names=_feature_names,
                                                   method='structure')

    print(
        f"feature_importance= {tree_sim__feature_importance}, predictions= {tree_sim__predictions}, structure= {tree_sim__structure}")

    return tree_sim__feature_importance, tree_sim__predictions, tree_sim__structure


def plot_time_series_with_stability(
        a_time_series: List[Tuple[Union[int, float], Union[List[float], np.ndarray]]],
        a_k: int,
        a_stability_df: Optional[pd.DataFrame] = None,
        decision_trees: Optional[List] = None,
        X_background: Optional[pd.DataFrame] = None,
        smoothing_window: int = 1,
        detected_change_points: Optional[List[int]] = None
) -> None:
    """
    Plots the time series with the interpolated and optionally smoothed stability values.

    Parameters:
    - a_time_series: time series data.
    - a_k: Decides on smoothing the data for sample counts
    - a_stability_df: DataFrame containing the stability values for each window.
    - smoothing_window: The window size for moving average smoothing of stability values. Default is 1 (no smoothing).
    - detected_change_points: List of timestamps where detected change points exist. Default is None (no detected change points plotted).
    """

    a_sample_counts = [Counter(sample_type for _, sample_type, time in a_time_series[i:i + a_k]) for i in
                       range(0, len(a_time_series),
                             a_k)]  # List of dictionaries containing the counts of samples for each timestamp.
    a_timestamps = range(0, len(a_time_series), a_k)

    plt.figure(figsize=(20, 12))

    # Plotting the original lines
    plt.plot(a_timestamps, [count['N'] for count in a_sample_counts], label='Normal')
    plt.plot(a_timestamps, [count['A1'] for count in a_sample_counts], label='Anomaly 1')
    plt.plot(a_timestamps, [count['A2'] for count in a_sample_counts], label='Anomaly 2')

    max_N = max([count['N'] for count in a_sample_counts])

    # Handle stability values if provided
    if a_stability_df is not None:
        # Interpolate the stability values
        window_centers = (a_stability_df['start'] + a_stability_df['end']) / 2
        stability_values = a_stability_df['stability_for_a_window']

        # Using linear interpolation to get the stability values for each timestamp
        interpolated_stability = np.interp(a_timestamps, window_centers, stability_values)

        # Apply moving average smoothing if smoothing_window > 1
        if smoothing_window > 1:
            interpolated_stability = pd.Series(interpolated_stability).rolling(window=smoothing_window, center=True,
                                                                               min_periods=1).mean().to_numpy()

        # Plotting the interpolated stability
        plt.plot(a_timestamps, interpolated_stability * 10 * max_N, label='Stability (interpolated)', linestyle='--',
                 color='grey')

    # Add vertical lines
    N1_effective = (N1 + N2) / 2
    print(f"N1_effective= {N1_effective}")
    plt.axvline(x=N1_effective, color='r', linestyle='--')
    plt.axvline(x=N1 + N2, color='g', linestyle='--')
    plt.axvline(x=N1 + N2 + N123, color='b', linestyle='--')
    plt.axvline(x=N1 + N2 + N123 + N3, color='y', linestyle='--')  # Line for N3

    # Create custom lines for the legend
    custom_lines = [
        Line2D([0], [0], color='r', lw=2, linestyle='--'),
        Line2D([0], [0], color='g', lw=2, linestyle='--'),
        Line2D([0], [0], color='b', lw=2, linestyle='--'),
        Line2D([0], [0], color='y', lw=2, linestyle='--'),
    ]

    # If detected_change_points is provided, add vertical lines for these points
    if detected_change_points is not None:
        for pt in detected_change_points:
            plt.axvline(x=pt, color='magenta', linestyle=':')
        # Adding a custom line for the legend
        custom_lines.append(Line2D([0], [0], color='magenta', lw=2, linestyle=':'))

    # Handle decision tree similarities if provided
    if decision_trees is not None:
        tree_sims_feature_importance = []
        tree_sims_predictions = []
        tree_sims_structure = []
        centers = []

        for i in range(len(decision_trees) - 1):
            _start = decision_trees[i][0]
            _end = decision_trees[i][1]
            _center = (_end + _start) / 3

            tree1 = decision_trees[i][3]
            tree2 = decision_trees[i + 1][3]

            sim_feature_importance, sim_predictions, sim_structure = get_decision_tree_similarities(tree1, tree2,
                                                                                                    X_background)
            tree_sims_feature_importance.append(sim_feature_importance)
            tree_sims_predictions.append(sim_predictions)
            tree_sims_structure.append(sim_structure)
            centers.append(_center)

        interpolated_feature_importance = np.interp(a_timestamps, centers, tree_sims_feature_importance)
        interpolated_predictions = np.interp(a_timestamps, centers, tree_sims_predictions)
        interpolated_structure = np.interp(a_timestamps, centers, tree_sims_structure)

        plt.plot(a_timestamps, interpolated_feature_importance * 10, label='Similarity (Feature Importance)',
                 linestyle='--', color='cyan')
        plt.plot(a_timestamps, interpolated_predictions * 10, label='Similarity (Predictions)', linestyle='--',
                 color='purple')
        plt.plot(a_timestamps, interpolated_structure * 10, label='Similarity (Structure)', linestyle='--',
                 color='orange')
        custom_lines.append(Line2D([0], [0], color='cyan', lw=2, linestyle='--'))
        custom_lines.append(Line2D([0], [0], color='purple', lw=2, linestyle='--'))
        custom_lines.append(Line2D([0], [0], color='orange', lw=2, linestyle='--'))

        # Create legends
    legend1 = plt.legend([line1, line2, line3], ['Normal', 'Anomaly 1', 'Anomaly 2'], loc="upper left")

    legend_labels = [
        'End of N1 (1.5k)',
        'End of N2 (1.5k)',
        'End of N123 (3.0k)',
        'End of N3 (2.0k)'
    ]
    if detected_change_points is not None:
        legend_labels.append('Detected Change Points')

    if decision_trees is not None:
        legend_labels.append('Similarity: Feature Importance')
        legend_labels.append('Similarity: Predictions')
        legend_labels.append('Similarity: Structure')

    legend2 = plt.legend(custom_lines, legend_labels, loc="upper right")

    plt.xlabel('Time')
    plt.ylabel('Sample Count')
    plt.title('Sample Type over Time with Stability')
    # Add the first legend manually to the current Axes
    plt.gca().add_artist(legend1)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


## OUTER STABILITY ###

def compare_explainer_between_windows(explainer_df: pd.DataFrame, window1: Tuple, window2: Tuple) -> pd.DataFrame:
    """
    Compare SHAP values for common observations between two windows.

    Parameters:
    - explainer_df: DataFrame containing SHAP values for every observation, could be interpolated
    - window1: Tuple representing the first window (start, end)
    - window2: Tuple representing the second window (start, end)

    Returns:
    - DataFrame with differences in SHAP values for common observations
    """
    # Extract data for each window
    data_window1 = explainer_df[(explainer_df['start'] == window1[0]) & (explainer_df['end'] == window1[1])]
    data_window2 = explainer_df[(explainer_df['start'] == window2[0]) & (explainer_df['end'] == window2[1])]

    # Find common observations
    common_obs = set(data_window1['observation_number']).intersection(set(data_window2['observation_number']))

    # Filter data for common observations
    common_data_window1 = data_window1[data_window1['observation_number'].isin(common_obs)]
    common_data_window2 = data_window2[data_window2['observation_number'].isin(common_obs)]

    # Calculate SHAP differences
    merged_data = pd.merge(common_data_window1, common_data_window2, on='observation_number', suffixes=('_w1', '_w2'))
    for column in ['sepal_length_cm_importance', 'sepal_width_cm_importance', 'petal_length_cm_importance',
                   'petal_width_cm_importance']:
        merged_data[f'{column}_diff'] = np.abs(merged_data[f'{column}_w1'] - merged_data[f'{column}_w2'])

    return merged_data


def aggregate_explainer_differences(interpolated_explainer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SHAP differences for all windows.

    Parameters:
    - interpolated_explainer_df: DataFrame containing interpolated SHAP values

    Returns:
    - DataFrame with aggregated SHAP differences for each window
    """
    all_windows = interpolated_explainer_df[['start', 'end']].drop_duplicates().values.tolist()
    all_diffs = []

    for i in range(len(all_windows)):
        for j in range(i + 1, len(all_windows)):
            window1 = all_windows[i]
            window2 = all_windows[j]

            # Calculate SHAP differences using the previously defined function
            difference_df = compare_explainer_between_windows(interpolated_explainer_df, tuple(window1), tuple(window2))

            # Calculate mean differences for each feature and overall
            mean_diffs = difference_df[
                ['sepal_length_cm_importance_diff', 'sepal_width_cm_importance_diff', 'petal_length_cm_importance_diff',
                 'petal_width_cm_importance_diff']].mean()
            overall_mean_diff = mean_diffs.mean()

            all_diffs.append(
                (window1[0], window1[1], window2[0], window2[1]) + tuple(mean_diffs) + (overall_mean_diff,))

    columns = ['start_w1', 'end_w1', 'start_w2', 'end_w2', 'sepal_length_cm_mean_diff', 'sepal_width_cm_mean_diff',
               'petal_length_cm_mean_diff', 'petal_width_cm_mean_diff', 'overall_mean_diff']
    return pd.DataFrame(all_diffs, columns=columns)


def plot_explainer_differences(aggregated_diffs_df: pd.DataFrame):
    """
    Plot the aggregated SHAP differences as a scatter plot.

    Parameters:
    - aggregated_diffs_df: DataFrame with aggregated SHAP differences for each window
    """
    plt.figure(figsize=(14, 8))

    # Taking midpoint of windows for x-axis
    x = (aggregated_diffs_df['start_w1'] + aggregated_diffs_df['end_w1']) / 2
    y = aggregated_diffs_df['overall_mean_diff']

    plt.scatter(x, y, c='blue', label='Mean SHAP Difference', alpha=0.6)
    plt.xlabel('Midpoint of Window 1')
    plt.ylabel('Mean SHAP Difference')
    plt.title('Aggregated SHAP Differences Between Windows')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_time_series_with_difference(
        a_time_series: List[Tuple[Union[int, float], Union[List[float], np.ndarray]]],
        a_k: int,
        a_aggregated_diffs_df: pd.DataFrame,
        smoothing_window: int = 1
) -> None:
    """
    Plots the time series with the interpolated and optionally smoothed aggregated SHAP differences.

    Parameters:
    - a_time_series: time series data.
    - a_k: Decides on smoothing the data for sample counts
    - a_aggregated_diffs_df: DataFrame containing the aggregated SHAP differences.
    - smoothing_window: The window size for moving average smoothing of aggregated SHAP differences. Default is 1 (no smoothing).
    """

    a_sample_counts = [Counter(sample_type for _, sample_type, time in a_time_series[i:i + a_k]) for i in
                       range(0, len(a_time_series), a_k)]
    a_timestamps = range(0, len(a_time_series), a_k)

    # Interpolate the aggregated differences
    window_centers_diff = (a_aggregated_diffs_df['start_w1'] + a_aggregated_diffs_df['end_w1']) / 2
    shap_diff_values = 1000 * a_aggregated_diffs_df['overall_mean_diff']

    # Apply moving average smoothing if smoothing_window > 1
    if smoothing_window > 1:
        shap_diff_values = pd.Series(shap_diff_values).rolling(window=smoothing_window, center=True,
                                                               min_periods=1).mean().to_numpy()

    fig, ax1 = plt.subplots(figsize=(20, 12))

    # Plotting the original lines on ax1
    ax1.plot(a_timestamps, [count['N'] for count in a_sample_counts], label='Normal')
    ax1.plot(a_timestamps, [count['A1'] for count in a_sample_counts], label='Anomaly 1')
    ax1.plot(a_timestamps, [count['A2'] for count in a_sample_counts], label='Anomaly 2')
    ax1.set_xlabel('Midpoint of Window 1')
    ax1.set_ylabel('Sample Count')
    ax1.legend(loc="upper left")

    # Plotting the smoothed SHAP differences on the same axis
    ax1.plot(window_centers_diff, shap_diff_values, label='SHAP Difference (interpolated and smoothed)', linestyle='-.',
             color='black')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


## Comparing decision trees ###
# Comparing decision trees can be approached in several ways, depending on what we mean by "similarity" or "stability":
#
# [structure] -> 1. Tree Structure Similarity: Examine the similarity of the structure of the two trees. This means checking if nodes are split using the same features and similar thresholds. This is the most direct form of comparison but can be cumbersome and sensitive to slight changes.
# Traverse both trees node by node and compare if the same features are used for splitting and if the thresholds are approximately similar. This will give a similarity ratio based on matched nodes.
#
# 2. Leaf Node Comparison: Compare the outcomes at the leaf nodes of both trees. If both trees classify instances similarly at their terminal nodes, they might be considered similar.
#
# [predictions] -> 3. Similarity via Predictions on a Test Set: Use a test set or background data to generate predictions from both trees. The similarity of their predictions can be measured using metrics such as accuracy, F1 score, etc. This method doesn't directly compare the trees but their outcomes.
#
# [feature_importance] -> 4. Feature Importance Similarity: Measure the importance of features in both trees. If both trees deem the same features as highly important, they may be considered similar.
# Compute the cosine similarity between the feature importance vectors of the two trees. Cosine similarity returns a value between 0 (completely dissimilar) and 1 (completely similar), which will represent our similarity measure.

def traverse_tree(tree, feature_names: list) -> list:
    """Utility function to traverse the decision tree and return tree structure."""
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    def recurse(node: int, depth: int) -> list:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left = recurse(tree_.children_left[node], depth + 1)
            right = recurse(tree_.children_right[node], depth + 1)
            return [(name, threshold, depth)] + left + right
        else:
            return []

    return recurse(0, 1)


def decision_tree_similarity(
        tree1: DecisionTreeClassifier,
        tree2: DecisionTreeClassifier,
        X_background: Optional[pd.DataFrame],
        feature_names: Optional[List[str]],
        method: str = 'predictions') -> float:
    """
    Compute the similarity between two decision trees.

    Parameters:
    - tree1: The first decision tree
    - tree2: The second decision tree
    - X_background: The background data for making predictions (used in 'predictions' method)
    - method: The method to use for comparison ('predictions', 'structure', 'feature_importance')
      - predictions: Similarity via Predictions on a Test Set: Use a test set or background data to generate predictions from both trees. The similarity of their predictions can be measured using metrics such as accuracy (used here), F1 score, etc. This method doesn't directly compare the trees but their outcomes. # FIDELITY
      - structure: Tree Structure Similarity: Examine the similarity of the structure of the two trees. This means checking if nodes are split using the same features and similar thresholds. This is the most direct form of comparison but can be cumbersome and sensitive to slight changes. Traverse both trees node by node and compare if the same features are used for splitting and if the thresholds are approximately similar. This will give a similarity ratio based on matched nodes.
      - feature_importance: Feature Importance Similarity: Measure the importance of features in both trees. If both trees deem the same features as highly important, they may be considered similar. Compute the cosine similarity between the feature importance vectors of the two trees. Cosine similarity returns a value between 0 (completely dissimilar) and 1 (completely similar), which will represent our similarity measure.

    Returns:
    - Similarity measure
    """

    _feature_names = feature_names or list(X_background.columns)

    if method == 'predictions':
        predictions_tree1 = tree1.predict(X_background)
        predictions_tree2 = tree2.predict(X_background)

        # ensemble_predictions = []
        matching_predictions = 0
        for p1, p2 in zip(predictions_tree1, predictions_tree2):
            if p1 == p2:
                # ensemble_predictions.append(p1)
                matching_predictions += 1
            else:
                # ensemble_predictions.append(y_background.mode().iloc[0, 0])
                pass

        # Convert y_background to numpy array for accuracy computation
        # y_background_array = y_background.values.ravel()
        # similarity_score = accuracy_score(y_background_array, ensemble_predictions)
        similarity_score = matching_predictions / len(predictions_tree1)

    elif method == 'structure':
        structure1 = set(traverse_tree(tree1, _feature_names))
        structure2 = set(traverse_tree(tree2, _feature_names))

        common_elements = structure1.intersection(structure2)
        total_elements = structure1.union(structure2)
        total_elements_len = len(total_elements)

        if total_elements_len > 0:
            similarity_score = len(common_elements) / total_elements_len
        else:
            print(f"Got empty structures!")
            similarity_score = 0.0

    elif method == 'feature_importance':
        importance1 = tree1.feature_importances_.reshape(1, -1)
        importance2 = tree2.feature_importances_.reshape(1, -1)

        similarity_score = cosine_similarity(importance1, importance2)[0][0]

    else:
        raise ValueError("Invalid method provided. Choose between 'predictions', 'structure', or 'feature_importance'.")

    return similarity_score

########################################################################################################################
## LOAD IRIS ###
iris = datasets.load_iris()
# print(iris.keys())

iris_df = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target']
)

iris_df['species'] = iris_df['target'].apply(
    lambda x: 'setosa' if x == 0 else ('versicolor' if x == 1 else 'virginica'))
iris_df['species'].value_counts()

## 1. "noise injection" / "noisy data augmentation".
NORMAL_CLASS = 'versicolor'
ANOMALY_CLASSES = ['setosa', 'virginica']
ANOMALY_CLASS_1 = ANOMALY_CLASSES[0]
ANOMALY_CLASS_2 = ANOMALY_CLASSES[1]
print(f"ANOMALY_CLASS_1 is {ANOMALY_CLASS_1}")
print(f"ANOMALY_CLASS_2 is {ANOMALY_CLASS_2}")

N = 1_000
np.random.seed(2023)
gauss_df = pd.DataFrame(np.resize(np.random.normal(loc=0.0, scale=.1, size=N * 4), (N, 4)))
# gauss_df  # N(0,1) leads to negative values

normal_gauss_df = iris_df[iris_df['species'] == NORMAL_CLASS].sample(n=N, replace=True, random_state=2023)
normal_gauss_df['sepal length (cm)'] += gauss_df[0].tolist()
normal_gauss_df['sepal width (cm)'] += gauss_df[1].tolist()
normal_gauss_df['petal length (cm)'] += gauss_df[2].tolist()
normal_gauss_df['petal width (cm)'] += gauss_df[3].tolist()
# get_stats(normal_gauss_df, 'species')

N2 = 1_000
np.random.seed(20232)
gauss_df_2 = pd.DataFrame(np.resize(np.random.normal(loc=0.0, scale=.1, size=N2 * 4), (N2, 4)))

anomaly_1_gauss_df_all = iris_df[iris_df['species'] == ANOMALY_CLASS_1].sample(n=N2, replace=True, random_state=20232)
anomaly_1_gauss_df_all['sepal length (cm)'] += gauss_df_2[0].tolist()
anomaly_1_gauss_df_all['sepal width (cm)'] += gauss_df_2[1].tolist()
anomaly_1_gauss_df_all['petal length (cm)'] += gauss_df_2[2].tolist()
anomaly_1_gauss_df_all['petal width (cm)'] += gauss_df_2[3].tolist()
# anomaly_1_gauss_df = anomaly_1_gauss_df_all.iloc[:int(N2 / 2)]
# anomaly_1_gauss_df_for_contrast_learn = anomaly_1_gauss_df_all.iloc[int(N2 / 2):]  # NOT USED
# print(get_stats(anomaly_1_gauss_df, 'species'))
# print(get_stats(anomaly_1_gauss_df_for_contrast_learn, 'species'))
filtered_df = anomaly_1_gauss_df_all[
    (anomaly_1_gauss_df_all['sepal length (cm)'] > 0) &
    (anomaly_1_gauss_df_all['sepal width (cm)'] > 0) &
    (anomaly_1_gauss_df_all['petal length (cm)'] > 0) &
    (anomaly_1_gauss_df_all['petal width (cm)'] > 0)
    ]
anomaly_1_gauss_df = filtered_df.sample(n=500, replace=False)

np.random.seed(20233)
gauss_df_3 = pd.DataFrame(np.resize(np.random.normal(loc=0.0, scale=.1, size=int(N2 * 4 / 2)), (int(N2 / 2), 4)))

anomaly_2_gauss_df = iris_df[iris_df['species'] == ANOMALY_CLASS_2].sample(n=int(N2 / 2), replace=True,
                                                                           random_state=20233)
anomaly_2_gauss_df['sepal length (cm)'] += gauss_df_3[0].tolist()
anomaly_2_gauss_df['sepal width (cm)'] += gauss_df_3[1].tolist()
anomaly_2_gauss_df['petal length (cm)'] += gauss_df_3[2].tolist()
anomaly_2_gauss_df['petal width (cm)'] += gauss_df_3[3].tolist()
# get_stats(anomaly_2_gauss_df, 'species')

### Create synthetic anomaly for contrastive learning
#  1. anomaly == normal + high gaus
#  2. anomaly == only gaus
#  3. anomaly == only uniform
# Using 3.:
N3 = 500
np.random.seed(20234)
uniform_df_3 = pd.DataFrame(np.random.uniform(low=0.0, high=10.0, size=(N3, 4)),
                            columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

## 2. Contrastive train autoencoder on normal and random data as contrast
normal_gauss_df_for_contrast_learn = normal_gauss_df.iloc[:int(N / 2)]
# print(f"anomaly_1_gauss_df_for_contrast_learn # = {len(anomaly_1_gauss_df_for_contrast_learn.index)}")
# print(f"normal_gauss_df_for_contrast_learn # = {len(normal_gauss_df_for_contrast_learn.index)}")  # NOT USED
# print(f"uniform_df_3 # = {len(uniform_df_3.index)}")


# Normalization
# anomaly_df = anomaly_1_gauss_df_for_contrast_learn  # Not used
normal_train_df = normal_gauss_df_for_contrast_learn

# scaler only for class Normal:
scaler = StandardScaler()
# scaler.fit(combined_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].to_numpy())
scaler.fit(
    normal_train_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].to_numpy())

anomaly_train_scaled = scaler.transform(uniform_df_3[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                                      'petal width (cm)']].to_numpy())  # WE USE INIFORM
normal_train_scaled = scaler.transform(
    normal_train_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].to_numpy())

anomaly_train_scaled_pt = torch.tensor(anomaly_train_scaled, dtype=torch.float32)
normal_train_scaled_pt = torch.tensor(normal_train_scaled, dtype=torch.float32)


## 15/12/2023 12:50 ###
