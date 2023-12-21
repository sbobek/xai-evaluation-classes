import hashlib
import os
import pickle
import random
from itertools import combinations
from typing import Any, Callable, List, Dict
from typing import Optional
from typing import Tuple, Union

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import load
from scipy.spatial.distance import euclidean
from scipy.stats import kstest, shapiro, skewnorm
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm.notebook import tqdm

np.int = int  # Fix deprecated

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


def get_losses_impl(X_tensor: torch.Tensor, model: nn.Module, label: str = '', do_print: bool = False) -> Tuple[
    List[float], float]:
    losses = []
    _criterion = nn.MSELoss()
    with torch.no_grad():
        for sample in X_tensor:
            output = model(sample)
            loss = _criterion(output, sample)
            losses.append(loss.item())

    # Choose a cutoff value
    cutoff = sum(losses) / len(losses)
    if do_print:
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
                 scaler: StandardScaler) -> None:
        self.autoencoder_model = autoencoder_model
        self.get_losses = get_losses
        self.loss_threshold = loss_threshold
        self.scaler = scaler
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


### Stability
def mock_explanation_function(X_df: pd.DataFrame, y_df: pd.DataFrame, index: int) -> List[List[Dict[str, Any]]]:
    columns = X_df.columns
    random_rule = lambda column: (
        column, f"{random.choice(['>=', '<'])}{random.uniform(X_df[column].min(), X_df[column].max())}")

    # Losowe wybieranie kolumn do utworzenia reguł
    selected_columns = random.sample(list(columns), random.randint(1, len(columns)))

    rule = {col: [random_rule(col)[1]] for col in selected_columns}

    return [[{
        'rule': rule,
        'prediction': str(random.randint(0, 1)),
        'confidence': random.uniform(0.5, 1.0)
    }]]


def parse_rule(rule: str) -> tuple:
    """Parses a rule into an operator and a value."""
    operator = rule[:2] if rule[1] in "=<" else rule[0]
    value = float(rule[2:]) if rule[1] in "=<" else float(rule[1:])
    return operator, value


def is_overlap(rule1: str, rule2: str, low: float = 0.0, high: float = 10.0) -> bool:
    """Checks if two rules have a partial overlap with given column boundaries."""
    op1, val1 = parse_rule(rule1)
    op2, val2 = parse_rule(rule2)

    # Dla reguł '>=', '>':
    if op1 in ['>=', '>'] and op2 in ['>=', '>']:
        return max(val1, val2) < high

    # Dla reguł '<=', '<':
    elif op1 in ['<=', '<'] and op2 in ['<=', '<']:
        return min(val1, val2) > low

    # Dla reguł mieszanych '>=', '<=' oraz '>', '<':
    elif op1 in ['>=', '>'] and op2 in ['<=', '<']:
        return val1 < val2 if op2 == '<' else val1 <= val2

    elif op1 in ['<=', '<'] and op2 in ['>=', '>']:
        return val1 > val2 if op2 == '>' else val1 >= val2

    return False


def calculate_overlap_degree(rules1: list, rules2: list, low: float = 0.0, high: float = 10.0) -> float:
    """
    Calculates the degree of overlap between two sets of rules.

    Each set of rules can contain one or two rules, defining a range.
    If a set contains one rule, the default boundaries (low and high) are used to define the range.
    If a set contains two rules, they are interpreted as lower and upper bounds of the range.
    The function returns the proportion of the overlap length to the average range length.

    Args:
    rules1 (list): The first set of rules.
    rules2 (list): The second set of rules.
    low (float): The default lower boundary if only one rule is provided.
    high (float): The default upper boundary if only one rule is provided.

    Returns:
    float: The degree of overlap between the two sets of rules.
    """

    def get_range(rules: list) -> tuple:
        """Determines the range defined by the rules."""
        assert len(rules) <= 2, "Each rule set must contain at most two rules."

        if len(rules) == 1:
            op, val = parse_rule(rules[0])
            return (max(val, low) if op in ['>=', '>'] else low, min(val, high) if op in ['<=', '<'] else high)
        else:
            op1, val1 = parse_rule(rules[0])
            op2, val2 = parse_rule(rules[1])
            lower_bound = max(val1, low) if op1 in ['>=', '>'] else max(val2, low)
            upper_bound = min(val1, high) if op1 in ['<=', '<'] else min(val2, high)
            return (lower_bound, upper_bound)

    range1 = get_range(rules1)
    range2 = get_range(rules2)

    # Calculating the overlap
    overlap_start = max(range1[0], range2[0])
    overlap_end = min(range1[1], range2[1])

    # Calculating the degree of overlap
    if overlap_start < overlap_end:
        overlap_length = overlap_end - overlap_start
        average_range_length = (range1[1] - range1[0] + range2[1] - range2[0]) / 2
        return overlap_length / average_range_length
    else:
        return 0.0  # No overlap


def calculate_jaccard_distance(explanation_i: List[Dict], explanation_j: List[Dict]) -> float:
    """
    Calculates the Jaccard distance between two explanations using the degree of overlap.

    The function computes the overlap degree for each column present in both explanations.
    If a column is present in one explanation but not in the other, its contribution to the metric is zero.
    Columns not present in either explanation are ignored.

    Args:
    explanation_1 (List[Dict]): The first explanation.
    explanation_2 (List[Dict]): The second explanation.

    Returns:
    float: The Jaccard distance between the two explanations.
    """

    def create_rule_set(explanation: List[Dict]) -> Dict[str, list]:
        rule_set = {}
        for exp in explanation:
            for col, rules in exp[0]['rule'].items():
                rule_set[col] = rules
        return rule_set

    set_1 = create_rule_set(explanation_i)
    set_2 = create_rule_set(explanation_j)

    # Calculating overlaps
    total_overlap = 0
    common_columns = set(set_1.keys()).intersection(set(set_2.keys()))
    for col in common_columns:
        overlap_degree = calculate_overlap_degree(set_1[col], set_2[col], 0.0, 10.0)
        total_overlap += overlap_degree

    # Calculating Jaccard distance
    total_columns = len(set(set_1.keys()).union(set(set_2.keys())))
    if total_columns == 0:
        return 0.0
    jaccard_distance = 1 - total_overlap / total_columns
    return jaccard_distance


def hash_dataframe(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


class ExplanationCache:
    def __init__(self):
        self.cache = {}

    def get_explanation(self, X_df: pd.DataFrame, y_df: pd.DataFrame, index: int, explanation_func: Callable) -> List[
        Dict]:
        X_hash = hash_dataframe(X_df)
        y_hash = hash_dataframe(y_df)
        key = (X_hash, y_hash, index)

        if key not in self.cache:
            self.cache[key] = explanation_func(X_df, y_df, index)
        return self.cache[key]

    def invalidate(self):
        self.cache.clear()


def calculate_stability(X_df: pd.DataFrame, y_df: pd.DataFrame, explanation_func: Callable, cache: ExplanationCache,
                        convergence_threshold: float = 0.001, min_iterations: int = 1_000,
                        max_iterations: int = 10_000, do_print: bool = False) -> (float, int):
    """
    Calculates the stability of explanations for a machine learning model.

    Stability is measured using local Lipschitz continuity in a fixed neighborhood of any datapoint.
    This approach evaluates how similar the explanations are for similar inputs.
    A lower value indicates higher stability, implying that small changes in input do not significantly alter the explanation.

    The stability metric is calculated as the maximum Lipschitz quotient across all pairs of data points.
    The Lipschitz quotient for a pair of points is the weighted Jaccard distance between their explanations divided by the Euclidean distance between the points.
    The Jaccard distance measures the dissimilarity between the sets of rules in the explanations.
    The weighting factor is the average of the 'confidence' values associated with each explanation, reflecting the certainty of the model in its predictions.

    Args:
    X_df (pd.DataFrame): The feature dataframe.
    y_df (pd.DataFrame): The target dataframe.
    explanation_func (Callable): The function used to generate explanations.
    cache (ExplanationCache): Cache object to store explanations.
    convergence_threshold (float): The threshold for convergence of stability calculation.
    min_iterations (int): The minimum number of iterations to perform.
    max_iterations (int): The maximum number of iterations to perform.
    do_print (bool): ...

    Returns:
    Tuple[float, int]: The calculated stability value and number of iterations.
    """
    max_lipschitz, prev_max_lipschitz = 0.0, None
    all_indices = list(combinations(range(len(X_df)), 2))
    random.shuffle(all_indices)
    current_it = 0

    with tqdm(total=max_iterations, desc="Calculating Stability") as pbar:
        while (prev_max_lipschitz is None or max_lipschitz <= 0.0 or abs(max_lipschitz - prev_max_lipschitz) > convergence_threshold or min_iterations > current_it) and current_it < max_iterations:
            prev_max_lipschitz = max_lipschitz
            # Sample a subset of indices for this iteration
            indices_subset = all_indices[current_it::max_iterations]
            for i, j in indices_subset:
                x_i, x_j = X_df.iloc[i], X_df.iloc[j]
                explanation_i = cache.get_explanation(X_df, y_df, i, explanation_func)
                explanation_j = cache.get_explanation(X_df, y_df, j, explanation_func)

                # Calculate Euclidean distance between x_i and x_j
                euclidean_distance = euclidean(x_i, x_j)

                # Calculate Jaccard distance between explanations using our custom function
                jaccard_distance = calculate_jaccard_distance(explanation_i, explanation_j)

                # Weight by confidence
                confidence_i = explanation_i[0][0]['confidence']
                confidence_j = explanation_j[0][0]['confidence']
                weighted_jaccard = jaccard_distance * (confidence_i + confidence_j) / 2

                # Calculate Lipschitz quotient
                lipschitz = weighted_jaccard / euclidean_distance if euclidean_distance != 0 else 0
                max_lipschitz = max(max_lipschitz, lipschitz)

                if do_print:
                    print("explanation_i", explanation_i)
                    print("explanation_j", explanation_j)
                    print("jaccard_distance", jaccard_distance, "weighted_jaccard", weighted_jaccard)
                    print("lipschitz", lipschitz, "max_lipschitz", max_lipschitz)

            current_it += 1
            pbar.update(1)

    return max_lipschitz, current_it

## 21/12/2023 18:18 ###
