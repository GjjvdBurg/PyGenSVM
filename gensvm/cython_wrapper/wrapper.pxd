cimport numpy as np

# Includes

cdef extern from "gensvm_globals.h":
    # Stuff for kerneltype
    ctypedef enum KernelType:
        pass

cdef extern from "gensvm_sparse.h":
    # stuff for GenSparse

    cdef struct GenSparse:
        long nnz
        long n_row
        long n_col
        double *values
        long *ia
        long *ja

    GenSparse *gensvm_init_sparse()
    void gensvm_free_sparse(GenSparse *)

cdef extern from "gensvm_base.h":

    cdef struct GenData:
        long K
        long n
        long m
        long r
        long *y
        double *Z
        GenSparse *spZ
        double *RAW
        double *Sigma
        KernelType kerneltype
        double *kernelparam

    cdef struct GenModel:
        int weight_idx
        long K
        long n
        long m
        double epsilon
        double p
        double kappa
        double lmd
        double *V
        double *Vbar
        double *U
        double *UU
        double *Q
        double *H
        double *rho
        double training_error
        KernelType kerneltype
        double *kernelparam
        double kernel_eigen_cutoff

    GenModel *gensvm_init_model()
    void gensvm_free_model(GenModel *)

    GenData *gensvm_init_data()
    void gensvm_free_data(GenData *)


cdef extern from "gensvm_task.h":

    cdef struct GenTask:
        long ID
        long folds
        GenData *train_data
        GenData *test_data

        KernelType kerneltype
        int weight_idx
        double p
        double kappa
        double lmd
        double epsilon
        double gamma
        double coef
        double degree
        double max_iter

        double performance
        double duration
        long *predictions

    GenTask *gensvm_init_task()
    gensvm_free_task(GenTask *)

cdef extern from "gensvm_train.h":

    void gensvm_train(GenModel *, GenData *, GenModel *) nogil

cdef extern from "gensvm_sv.h":

    long gensvm_num_sv(GenModel *)

cdef extern from "gensvm_queue.h":

    cdef struct GenQueue:
        GenTask **tasks
        long N
        long i

    GenQueue *gensvm_init_queue()
    void gensvm_free_queue(GenQueue *)

cdef extern from "gensvm_helper.c":

    ctypedef char* char_const_ptr "char const *"
    void set_model(GenModel *, double, double, double, double, int, int, 
            double, double, double, double, long, long)
    void set_seed_model(GenModel *, double, double, double, double, int, int, 
            double, double, double, double, long, long, char *, long, long)
    void set_raw_weights(GenModel *, char *, int)
    void set_data(GenData *, char *, char *, np.npy_intp *, long)
    void set_task(GenTask *, int, GenData *, int, double, double, double, 
            double, double, int, double, double, double, long)
    char_const_ptr check_model(GenModel *)
    void copy_V(void *, GenModel *)
    long get_iter_count(GenModel *)
    double get_training_error(GenModel *)
    int get_status(GenModel *)
    long get_n(GenModel *)
    long get_m(GenModel *)
    long get_K(GenModel *)
    void free_data(GenData *)
    void set_verbosity(int)
    void gensvm_predict(char *, char *, long, long, long, char *) nogil
    void gensvm_predict_kernels(char *, char *, char *, long, long, long, 
            long, long, long, int, double,  double, double, double, char *) nogil
    void gensvm_train_q_helper(GenQueue *, char *, int, int) nogil
    void set_queue(GenQueue *, long, GenTask **)
    double get_task_duration(GenTask *)
    double get_task_performance(GenTask *)
    void copy_task_predictions(GenTask *, char *, long)
    void copy_task_durations(GenTask *, char *, int)
