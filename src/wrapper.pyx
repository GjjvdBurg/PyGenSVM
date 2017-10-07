"""
Wrapper for GenSVM

Not implemented yet:
    - vector of instance weights
    - class weights
    - seed model
    - max_iter = -1 for unlimited

"""

from __future__ import print_function

import numpy as np
cimport numpy as np

cimport wrapper

np.import_array()

GENSVM_KERNEL_TYPES = ["linear", "poly", "rbf", "sigmoid"]

def train_wrap(
        np.ndarray[np.float64_t, ndim=2, mode='c'] X,
        np.ndarray[np.int_t, ndim=1, mode='c'] y,
        double p=1.0,
        double lmd=pow(2, -8),
        double kappa=0.0,
        double epsilon=1e-6,
        int weight_idx=1,
        str kernel='linear',
        double gamma=1.0,
        double coef=0.0,
        double degree=2.0,
        double kernel_eigen_cutoff=1e-8,
        int max_iter=100000000,
        int random_seed=-1):
    """
    """

    # Initialize model and data
    cdef GenModel *model = gensvm_init_model()
    cdef GenData *data = gensvm_init_data()
    cdef long n_obs
    cdef long n_var
    cdef long n_class

    # get the kernel index
    kernel_index = GENSVM_KERNEL_TYPES.index(kernel)

    # get the number of classes
    classes = np.unique(y)
    n_obs = X.shape[0]
    n_var = X.shape[1]
    n_class = classes.shape[0]

    # Set the data
    set_data(data, X.data, y.data, X.shape, n_class)

    # Set the model
    set_model(model, p, lmd, kappa, epsilon, weight_idx, kernel_index, degree, 
            gamma, coef, kernel_eigen_cutoff, max_iter, random_seed)

    # Check the parameters
    error_msg = check_model(model)
    if error_msg:
        gensvm_free_model(model)
        free_data(data)
        error_repl = error_msg.decode('utf-8')
        raise ValueError(error_repl)

    # Do the actual training
    with nogil:
        gensvm_train(model, data, NULL)

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

def set_verbosity_wrap(int verbosity):
    """
    Control verbosity of gensvm library
    """
    set_verbosity(verbosity)
