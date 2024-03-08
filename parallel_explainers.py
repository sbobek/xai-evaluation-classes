import os
from multiprocessing import Pool
from typing import Any

import numpy as np
import tensorflow as tf
from anchor import anchor_tabular
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import load_model
from tqdm import tqdm

np.int = int  # Fix deprecated for LUX


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # second gpu


class ModelWrapper:
    def __init__(self, model: Any, timesteps: int, features: int):
        self.model = model
        self.timesteps = timesteps
        self.features = features

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Reshape flattened data back to (n_windows, timesteps, features)
        if len(X.shape) == 1:  # Single sample
            n_samples = 1
            X_reshaped = X.reshape(n_samples, self.timesteps, self.features)
        else:  # Batch of samples
            # Calculate the number of samples by dividing the flat array's length by (timesteps * features)
            n_samples = X.shape[0]
            # Reshape the flat array to (n_samples, timesteps, features)
            X_reshaped = X.reshape(n_samples, self.timesteps, self.features)

        # Make predictions with the reshaped data
        preds = self.model.predict(X_reshaped)
        if preds.shape[1] == 1:  # Binary classification
            # For binary classification, return the complement probability for the negative class
            return np.hstack([1 - preds, preds])
        else:  # Multiclass classification
            return preds


def compute_single_lime_explanation(index, instance, model_wrapper, explainer,
                                    num_features, timesteps, flattened_feature_names):
    """
    Compute LIME explanation for a single instance and reshape its importance.

    :param index: Index of the instance for identification.
    :param instance: The instance to explain, expected to be in the original shape (timesteps, features).
    :param model_wrapper: The model wrapper with a predict_proba function.
    :param explainer: The LIME TabularExplainer object.
    :param num_features: The total number of features (timesteps * features per timestep).
    :param timesteps: The number of timesteps in the instance data.
    :return: A tuple of the index and the reshaped LIME importance.
    """
    # Flatten the instance for LIME
    flattened_instance = instance.reshape(1, -1)[0]

    # Get the LIME explanation for the flattened instance
    explanation = explainer.explain_instance(
        flattened_instance, model_wrapper.predict_proba, num_features=num_features * timesteps)

    # Get the LIME importance for the features and reshape it
    # print(explanation.local_exp)
    lime_importance = np.array([explanation.local_exp[1][i][1] for i in range(num_features * timesteps)])
    reshaped_importance = lime_importance.reshape((timesteps, num_features))

    return index, reshaped_importance


def compute_lime_explanation(instances, model_wrapper, explainer, num_features, timesteps):
    """
    Explain multiple instances using LIME and reshape their importance values.

    :param instances: The instances to explain, expected to be in the original shape (samples, timesteps, features).
    :param model_wrapper: The model wrapper with a predict_proba function.
    :param explainer: The LIME TabularExplainer object.
    :param num_features: The number of features per timestep in the instance data.
    :param timesteps: The number of timesteps in the instance data.
    :return: A 3D array with explanations reshaped to original instance shapes (samples, timesteps, features).
    """
    ordered_results = []
    for index, instance in enumerate(instances):
        _, reshaped_importance = compute_single_lime_explanation(
            index, instance, model_wrapper, explainer, num_features, timesteps, None)
        ordered_results.append(reshaped_importance)

    # Combine the explanations into a single array
    combined_explanations = np.array(ordered_results)
    return combined_explanations


def worker_init_lime(model_path, num_features, timesteps,
                     flattened_background_data, flattened_feature_names, class_names,
                     num_cores, gpu_idx):
    """
    Initialize worker by loading the model into a global variable.
    This function is called when each worker process starts.
    """
    global explainer
    explainer = LimeTabularExplainer(
        flattened_background_data,
        feature_names=flattened_feature_names,
        class_names=class_names,
        mode='classification')
    print("explainer initialised\n")

def worker_init_lime_and_model(model_path, num_features, timesteps,
                     flattened_background_data, flattened_feature_names, class_names,
                     num_cores, gpu_idx):
    worker_init_lime(model_path, num_features, timesteps,
                     flattened_background_data, flattened_feature_names, class_names,
                     num_cores, gpu_idx)
    worker_init_model(model_path, num_features, timesteps,
                      flattened_background_data, flattened_feature_names, class_names,
                      num_cores, gpu_idx)



def worker_init_model(model_path, num_features, timesteps,
                      flattened_background_data, flattened_feature_names, class_names,
                      num_cores, gpu_idx):
    """
    Initialize worker by loading the model into a global variable.
    This function is called when each worker process starts.
    """
    print(f"loading model {model_path}\n")
    # Ensure TensorFlow uses only necessary resources and does not conflict with others
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    global model_wrapper
    # from tensorflow.keras.models import load_model
    # from tensorflow.python import keras
    tf.keras.backend.clear_session()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    gpu_mem_mb = 24564
    gpus = tf.config.list_physical_devices("GPU")
    print("gpus", gpus)
    if len(gpus) > 0 and gpu_idx >= 0:
        if gpu_idx + 1 > len(gpus):
            gpu_idx = len(gpus) - 1
        gpu_to_limit = gpus[gpu_idx]
        tf.config.set_logical_device_configuration(
            gpu_to_limit, [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem_mb // (num_cores + 1))]
        )
    model_classifier = load_model(model_path)
    # model_classifier._make_predict_function()
    print("model_classifier loaded\n")
    e_model_compilation_params = {'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']}
    model_classifier.compile(**e_model_compilation_params)
    model_wrapper = ModelWrapper(model_classifier, timesteps=timesteps, features=num_features)
    print("model_wrapper initialised\n")


def compute_single_lime_explanation_parallel_helper(args):
    """
    Helper function to unpack arguments and call the actual function
    that computes the LIME explanation for a single instance.
    """
    # model_wrapper, explainer are global
    index, instance, num_features, timesteps = args
    _, reshaped_importance = compute_single_lime_explanation(
        index, instance, model_wrapper, explainer, num_features, timesteps, None)

    return reshaped_importance



def compute_lime_explanation_parallel(instances,
                                      model_path, num_features, timesteps,
                                      flattened_background_data, flattened_feature_names, class_names,
                                      num_cores, gpu_idx):
    """
    Explain multiple instances using LIME in parallel and reshape their importance values.

    :param instances: The instances to explain, expected to be in the original shape (samples, timesteps, features).
    :param num_features: The number of features per timestep in the instance data.
    :param timesteps: The number of timesteps in the instance data.
    :param num_cores: The number of CPU cores to use for parallel computation.
    :param gpu_idx: .

    :return: A 3D array with explanations reshaped to original instance shapes (samples, timesteps, features).
    """

    # Set up multiprocessing pool with the specified number of workers
    # and initialize each worker using the worker_init function

    pool = Pool(processes=num_cores, initializer=worker_init_lime_and_model, initargs=(
        model_path, num_features, timesteps,
        flattened_background_data, flattened_feature_names, class_names,
        num_cores, gpu_idx))

    # Prepare arguments for parallel computation
    args = [(i, instance, num_features, timesteps) for i, instance in enumerate(instances)]

    # Compute explanations in parallel
    ordered_results = pool.map(compute_single_lime_explanation_parallel_helper, args)

    # Shutdown the pool and free resources
    pool.close()
    pool.join()

    # Combine the explanations into a single array
    combined_explanations = np.array(ordered_results)
    return combined_explanations


def compute_single_anchor_explanation(index, instance, model_wrapper, explainer, num_features, timesteps,
                                      background_feature_names):
    fn_predict = lambda x: [np.argmax(pred) for pred in model_wrapper.predict_proba(x)]
    best_exp = explainer.explain_instance(
        instance,
        fn_predict,
        threshold=0.95)

    prediction = fn_predict(instance)[0]
    # def compute_single_anchor_explanation(index, best_exp, prediction, background_feature_names):
    # Convert AnchorExplanation to the specified format
    explanation = []
    if best_exp:
        rule_dict = {}
        for feature_name in background_feature_names:
            for rule in best_exp.names():
                # print("rule", rule)
                if feature_name in rule:
                    processed_rules = []
                    for op in ["> ", ">= "]:
                        if op in rule:
                            left_rules = rule.split(op)
                            if feature_name in left_rules[0]:
                                processed_rules.append((op + left_rules[1]).replace(' ', ''))
                            else:
                                assert feature_name in left_rules[1]
                                processed_rules.append(
                                    (("< " if op == "> " else "<= ") + left_rules[0]).replace(' ', ''))
                    for op in ["< ", "<= "]:
                        if op in rule:
                            right_rules = rule.split(op)
                            if feature_name in right_rules[1]:
                                processed_rules.append(
                                    (("> " if op == "< " else ">= ") + right_rules[0]).replace(' ', ''))
                            else:
                                assert feature_name in right_rules[0]
                                processed_rules.append((op + right_rules[1]).replace(' ', ''))
                    rule_dict[feature_name] = processed_rules

        explanation.append(
            [{'rule': rule_dict, 'prediction': str(int(prediction)), 'confidence': best_exp.precision()}])

    return index, explanation


def compute_anchor_explanation(instances, model_wrapper, explainer, num_features, timesteps,
                               background_feature_names):
    """
    Explain multiple instances using ANCHOR and reshape their importance values.

    :param instances: The instances to explain, expected to be in the original shape (samples, timesteps, features).
    :param model_wrapper: The model wrapper with a predict_proba function.
    :param explainer: The LIME TabularExplainer object.
    :param num_features: The number of features per timestep in the instance data.
    :param timesteps: The number of timesteps in the instance data.
    :param background_feature_names: .
    :return: .
    """
    instances_shape = instances.shape
    if len(instances_shape) == 3:
        instances = instances.reshape(instances_shape[0], -1)
        print("new instances.shape", instances.shape)

    ordered_results = []
    for index, instance in tqdm(enumerate(instances), total=len(instances)):
        _, reshaped_importance = compute_single_anchor_explanation(
            index, instance, model_wrapper, explainer, num_features, timesteps, background_feature_names)
        ordered_results.append(reshaped_importance)

    # Combine the explanations into a single array
    combined_explanations = np.array(ordered_results)
    return combined_explanations


def worker_init(model_path, num_features, timesteps,
                flattened_background_data, flattened_feature_names, class_names,
                worker_init_explainer_fn, worker_init_model_fn,
                num_cores, gpu_idx):
    worker_init_explainer_fn(model_path, num_features, timesteps,
                             flattened_background_data, flattened_feature_names, class_names,
                             num_cores, gpu_idx)
    worker_init_model_fn(model_path, num_features, timesteps,
                         flattened_background_data, flattened_feature_names, class_names,
                         num_cores, gpu_idx)


def compute_single_generic_explanation_parallel_helper(args):
    """
    Helper function to unpack arguments and call the actual function
    that computes the explanation for a single instance.
    """
    # model_wrapper, explainer are global
    index, instance, num_features, timesteps, flattened_feature_names, compute_single_generic_explanation_fn = args
    _, reshaped_importance = compute_single_generic_explanation_fn(
        index, instance, model_wrapper, explainer, num_features, timesteps, flattened_feature_names)

    return reshaped_importance


def compute_generic_explanation_parallel(instances,
                                         model_path, num_features, timesteps,
                                         flattened_background_data, flattened_feature_names, class_names,
                                         worker_init_explainer_fn, worker_init_model_fn,
                                         compute_single_generic_explanation_fn,
                                         num_cores, gpu_idx):
    """
    Explain multiple instances using LIME in parallel and reshape their importance values.

    :param instances: The instances to explain, expected to be in the original shape (samples, timesteps, features).
    :param num_features: The number of features per timestep in the instance data.
    :param timesteps: The number of timesteps in the instance data.
    :param num_cores: The number of CPU cores to use for parallel computation.
    :param gpu_idx: .

    :return: A 3D array with explanations reshaped to original instance shapes (samples, timesteps, features).
    """
    instances_shape = instances.shape
    if len(instances_shape) == 3:
        instances = instances.reshape(instances_shape[0], -1)
        print("new instances.shape", instances.shape)

    # Set up multiprocessing pool with the specified number of workers
    # and initialize each worker using the worker_init function

    pool = Pool(processes=num_cores, initializer=worker_init, initargs=(
        model_path, num_features, timesteps,
        flattened_background_data, flattened_feature_names, class_names,
        worker_init_explainer_fn, worker_init_model_fn,
        num_cores, gpu_idx))

    # Prepare arguments for parallel computation
    args = [(i, instance, num_features, timesteps,
             flattened_feature_names, compute_single_generic_explanation_fn
             ) for i, instance in enumerate(instances)]

    # Compute explanations in parallel
    ordered_results = pool.map(compute_single_generic_explanation_parallel_helper, args)

    # Shutdown the pool and free resources
    pool.close()
    pool.join()

    # Combine the explanations into a single array
    combined_explanations = np.array(ordered_results)
    return combined_explanations


def worker_init_anchor(model_path, num_features, timesteps,
                       flattened_background_data, flattened_feature_names, class_names,
                       num_cores, gpu_idx):
    """
    Initialize worker by loading the model into a global variable.
    This function is called when each worker process starts.
    """
    global explainer
    explainer = anchor_tabular.AnchorTabularExplainer(
        train_data=flattened_background_data,
        feature_names=flattened_feature_names,
        class_names=[0, 1],
    )
    print("explainer initialised\n")

### 08.03.2024: 09:47 ###
