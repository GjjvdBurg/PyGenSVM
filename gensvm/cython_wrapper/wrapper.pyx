# cython: language_level=2

"""
Wrapper for GenSVM

Not implemented yet:
    - vector of instance weights
    - class weights
    - seed model
    - max_iter = -1 for unlimited

"""

from __future__ import print_function

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

cimport wrapper

np.import_array()

GENSVM_KERNEL_TYPES = ["linear", "poly", "rbf", "sigmoid"]

def train_wrap(
        np.ndarray[np.float64_t, ndim=2, mode='c'] X,
        np.ndarray[np.int_t, ndim=1, mode='c'] y,
        long n_class,
        double p=1.0,
        double lmd=pow(2, -8),
        double kappa=0.0,
        double epsilon=1e-6,
        int weight_idx=1,
        np.ndarray[np.float64_t, ndim=1, mode='c'] raw_weights=None,
        str kernel='linear',
        double gamma=1.0,
        double coef=0.0,
        double degree=2.0,
        double kernel_eigen_cutoff=1e-8,
        int max_iter=100000000,
        int random_seed=-1,
        np.ndarray[np.float64_t, ndim=2, mode='c'] seed_V=None
        ):
    """
    """

    # Initialize model and data
    cdef GenModel *model = gensvm_init_model()
    cdef GenData *data = gensvm_init_data()
    cdef GenModel *seed_model = gensvm_init_model()
    cdef long n_obs
    cdef long n_var

    # get the kernel index
    kernel_index = GENSVM_KERNEL_TYPES.index(kernel)

    # get the number of classes
    n_obs = X.shape[0]
    n_var = X.shape[1]

    # Set the data
    set_data(data, X.data, y.data, X.shape, n_class)

    # Set the model
    set_model(model, p, lmd, kappa, epsilon, weight_idx, kernel_index, degree, 
            gamma, coef, kernel_eigen_cutoff, max_iter, random_seed)

    if not seed_V is None:
        set_seed_model(seed_model, p, lmd, kappa, epsilon, weight_idx, 
                kernel_index, degree, gamma, coef, kernel_eigen_cutoff, 
                max_iter, random_seed, seed_V.data, n_var, n_class)
    else:
        gensvm_free_model(seed_model)
        seed_model = NULL

    if not raw_weights is None:
        set_raw_weights(model, raw_weights.data, n_obs)

    # Check the parameters
    error_msg = check_model(model)
    if error_msg:
        gensvm_free_model(model)
        gensvm_free_model(seed_model)
        free_data(data)
        error_repl = error_msg.decode('utf-8')
        raise ValueError(error_repl)

    # Do the actual training
    with nogil:
        gensvm_train(model, data, seed_model)

    # update the number of variables (this may have changed due to kernel)
    n_var = get_m(model)

    # copy the results
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] V
    V = np.empty((n_var+1, n_class-1))
    copy_V(V.data, model)

    # get other results from model
    iter_count = get_iter_count(model)
    training_error = get_training_error(model)
    fit_status = get_status(model)
    n_SV = gensvm_num_sv(model)

    # free model and data
    gensvm_free_model(model);
    gensvm_free_model(seed_model)
    free_data(data);

    return (V, n_SV, iter_count, training_error, fit_status)


def predict_wrap(
        np.ndarray[np.float64_t, ndim=2, mode='c'] X,
        np.ndarray[np.float64_t, ndim=2, mode='c'] V
        ):
    """
    """

    cdef long n_test_obs
    cdef long n_var
    cdef long n_class

    n_test_obs = X.shape[0]
    n_var = X.shape[1]
    n_class = V.shape[1] + 1

    # output vector
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] predictions
    predictions = np.empty((n_test_obs, ), dtype=np.int)

    # do the prediction
    with nogil:
        gensvm_predict(X.data, V.data, n_test_obs, n_var, n_class, 
                predictions.data)

    return predictions


def predict_kernels_wrap(
        np.ndarray[np.float64_t, ndim=2, mode='c'] Xtest,
        np.ndarray[np.float64_t, ndim=2, mode='c'] Xtrain,
        np.ndarray[np.float64_t, ndim=2, mode='c'] V,
        long n_class,
        int kernel_idx,
        double gamma,
        double coef,
        double degree,
        double kernel_eigen_cutoff
        ):
    """
    Compute predictions for nonlinear GenSVM. Calls the C helper function 
    "gensvm_predict_kernels", which in turn calls the appropriate library 
    functions.
    """

    cdef long n_obs_test
    cdef long n_obs_train
    cdef long n_var
    cdef long V_rows = V.shape[0]
    cdef long V_cols = V.shape[1]

    n_obs_test = Xtest.shape[0]
    n_obs_train = Xtrain.shape[0]
    n_var = Xtrain.shape[1]

    cdef np.ndarray[np.int_t, ndim=1, mode='c'] predictions
    predictions = np.empty((n_obs_test, ), dtype=np.int)

    with nogil:
        gensvm_predict_kernels(Xtest.data, Xtrain.data, V.data, V_rows, 
                V_cols, n_obs_train, n_obs_test, n_var, n_class, kernel_idx, 
                gamma, coef, degree, kernel_eigen_cutoff, predictions.data)

    return predictions


def grid_wrap(
        np.ndarray[np.float64_t, ndim=2, mode='c'] X,
        np.ndarray[np.int_t, ndim=1, mode='c'] y,
        candidate_params,
        int store_predictions,
        np.ndarray[np.int_t, ndim=1, mode='c'] cv_idx,
        int n_folds,
        int verbosity,
        ):
    """
    """

    cdef GenQueue *queue = gensvm_init_queue()
    cdef GenData *data = gensvm_init_data()
    cdef GenTask *task
    cdef long n_obs
    cdef long n_var
    cdef long n_class
    cdef long n_tasks = len(candidate_params)

    # get the number of classes
    classes = np.unique(y)
    n_obs = X.shape[0]
    n_var = X.shape[1]
    n_class = classes.shape[0]

    set_data(data, X.data, y.data, X.shape, n_class)

    cdef GenTask **tasks = <GenTask **>malloc(n_tasks * sizeof(GenTask *))

    ID = 0
    for candidate in candidate_params:
        params = {
                'p': 1.0,
                'lmd': 1e-5,
                'kappa': 0.0,
                'epsilon': 1e-6,
                'weight_idx': 1,
                'kernel': GENSVM_KERNEL_TYPES.index('linear'),
                'gamma': 1.0,
                'coef': 0.0,
                'degree': 2.0,
                'max_iter': 1e8
                }
        params.update(candidate)
        if 'kernel' in candidate:
            params['kernel'] = GENSVM_KERNEL_TYPES.index(candidate['kernel'])
        if 'weights' in candidate:
            params['weight_idx'] = 1 if candidate['weights'] == 'unit' else 2

        task = gensvm_init_task()
        set_task(task, ID, data, n_folds, params['p'], params['lmd'], 
                params['kappa'], params['epsilon'], params['weight_idx'], 
                params['kernel'], params['degree'], params['gamma'], 
                params['coef'], params['max_iter'])

        tasks[ID] = task
        ID += 1

    set_queue(queue, n_tasks, tasks)

    with nogil:
        gensvm_train_q_helper(queue, cv_idx.data, store_predictions, 
                verbosity)

    cdef np.ndarray[np.int_t, ndim=1, mode='c'] pred
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] dur

    results = dict()
    results['params'] = []
    results['duration'] = []
    results['scores'] = []
    # predictions: for each task, an array of size n_obs with class 
    # predictions (-1 if missing)
    results['predictions'] = []
    # durations: for each task, an array of size n_folds with duration for 
    # each fold (nan if missing)
    results['durations'] = []
    for ID in range(n_tasks):
        results['params'].append(candidate_params[ID])
        results['scores'].append(get_task_performance(tasks[ID]))
        if store_predictions:
            pred = np.zeros((n_obs, ), dtype=np.int)
            copy_task_predictions(tasks[ID], pred.data, n_obs)
            results['predictions'].append(pred.copy())
        dur = np.zeros((n_folds, ), dtype=np.double)
        copy_task_durations(tasks[ID], dur.data, n_folds)
        results['durations'].append(dur.copy())

    gensvm_free_queue(queue)
    free_data(data)

    return results


def set_verbosity_wrap(int verbosity):
    """
    Control verbosity of gensvm library
    """
    set_verbosity(verbosity)
