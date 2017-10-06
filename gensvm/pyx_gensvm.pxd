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

cdef extern from "gensvm_train.h":

    void gensvm_train(GenModel *, GenData *, GenModel *) nogil

cdef extern from "gensvm_sv.h":

    long gensvm_num_sv(GenModel *)

cdef extern from "gensvm_helper.c":

    ctypedef char* char_const_ptr "char const *"
    void set_model(GenModel *, double, double, double, double, int, int, 
            double, double, double, double, long, long)
    void set_data(GenData *, char *, char *, np.npy_intp *, long)
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
