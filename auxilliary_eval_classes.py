import numpy as np
import pandas as pd

np.int = int  # Fix deprecated

import pickle
from joblib import load
import dill

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

from scipy.stats import kstest, shapiro, skewnorm

import torch
from torch import nn

from typing import Optional

from typing import Callable, List, Tuple, Union

from sklearn import datasets
import matplotlib.pyplot as plt

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


def get_losses_impl(X_tensor: torch.Tensor, model: nn.Module, label: str = '') -> Tuple[List[float], float]:
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


## CLASSIFIER ON AUTOENCODER ###

class AnomalyClassifier(BaseEstimator, ClassifierMixin):
    """
    Takes into account Right-skew distribution ("Rozkład prawoskośny")
    """

    def __init__(self,
                 autoencoder_model: torch.nn.Module,
                 get_losses: Callable[[Union[np.ndarray, torch.Tensor, pd.DataFrame], torch.nn.Module], List[float]],
                 loss_threshold: float,
                 scaler) -> None:
        self.autoencoder_model = autoencoder_model
        self.get_losses = get_losses
        self.loss_threshold = loss_threshold
        self.scaler = scaler
        self.a_scaler = None
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

normal_gauss_df = iris_df[iris_df['species'] == NORMAL_CLASS].sample(n=N, replace=True, random_state=2023)
normal_gauss_df['sepal length (cm)'] += gauss_df[0].tolist()
normal_gauss_df['sepal width (cm)'] += gauss_df[1].tolist()
normal_gauss_df['petal length (cm)'] += gauss_df[2].tolist()
normal_gauss_df['petal width (cm)'] += gauss_df[3].tolist()

N2 = 1_000
np.random.seed(20232)
gauss_df_2 = pd.DataFrame(np.resize(np.random.normal(loc=0.0, scale=.1, size=N2 * 4), (N2, 4)))

anomaly_1_gauss_df_all = iris_df[iris_df['species'] == ANOMALY_CLASS_1].sample(n=N2, replace=True, random_state=20232)
anomaly_1_gauss_df_all['sepal length (cm)'] += gauss_df_2[0].tolist()
anomaly_1_gauss_df_all['sepal width (cm)'] += gauss_df_2[1].tolist()
anomaly_1_gauss_df_all['petal length (cm)'] += gauss_df_2[2].tolist()
anomaly_1_gauss_df_all['petal width (cm)'] += gauss_df_2[3].tolist()

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

### Create synthetic anomaly for contrastive learning
#  3. anomaly == only uniform
N3 = 500
np.random.seed(20234)
uniform_df_3 = pd.DataFrame(np.random.uniform(low=0.0, high=10.0, size=(N3, 4)),
                            columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

## 2. Contrastive train autoencoder on normal and random data as contrast
normal_gauss_df_for_contrast_learn = normal_gauss_df.iloc[:int(N / 2)]

# Normalization
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

### 3. Test autoencoder, visualise losses and choose cutoff
X_normal_test_df = normal_gauss_df.iloc[int(N / 2):][
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].to_numpy()
X_anomaly_1_test_df = anomaly_1_gauss_df[
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].to_numpy()
X_anomaly_2_test_df = anomaly_2_gauss_df[
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].to_numpy()
print(f"X_normal # = {len(X_normal_test_df)}")
print(f"X_anomaly_1 # = {len(X_anomaly_1_test_df)}")
print(f"X_anomaly_2 # = {len(X_anomaly_2_test_df)}")


## 16/12/2023 20:51 ###
