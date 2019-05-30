
.. py:class:: GenSVM(p=1.0, lmd=1e-05, kappa=0.0, epsilon=1e-06, weights='unit', kernel='linear', gamma='auto', coef=1.0, degree=2.0, kernel_eigen_cutoff=1e-08, verbose=0, random_state=None, max_iter=100000000.0)
   :noindex:
   :module: gensvm.core

   Generalized Multiclass Support Vector Machine Classification.
   
   This class implements the basic GenSVM classifier. GenSVM is a generalized
   multiclass SVM which is flexible in the weighting of misclassification
   errors. It is this flexibility that makes it perform well on diverse
   datasets.
   
   The :func:`~GenSVM.fit` and :func:`~GenSVM.predict` methods of this class
   use the GenSVM C library for the actual computations.
   
   :param p: Parameter for the L_p norm of the loss function (1.0 <= p <= 2.0)
   :type p: float, optional (default=1.0)
   :param lmd: Parameter for the regularization term of the loss function (lmd > 0)
   :type lmd: float, optional (default=1e-5)
   :param kappa: Parameter for the hinge function in the loss function (kappa > -1.0)
   :type kappa: float, optional (default=0.0)
   :param weights: Type of sample weights to use. Options are 'unit' for unit weights and
                   'group' for group size correction weights (equation 4 in the paper).
   
                   It is also possible to provide an explicit vector of sample weights
                   through the :func:`~GenSVM.fit` method. If so, it will override the
                   setting provided here.
   :type weights: string, optional (default='unit')
   :param kernel: Specify the kernel type to use in the classifier. It must be one of
                  'linear', 'poly', 'rbf', or 'sigmoid'.
   :type kernel: string, optional (default='linear')
   :param gamma: Kernel parameter for the rbf, poly, and sigmoid kernel. If gamma is
                 'auto' then 1/n_features will be used. See `Kernels in GenSVM
                 <gensvm_kernels_>`_ for the exact implementation of the kernels.
   :type gamma: float, optional (default='auto')
   :param coef: Kernel parameter for the poly and sigmoid kernel. See `Kernels in
                GenSVM <gensvm_kernels_>`_ for the exact implementation of the kernels.
   :type coef: float, optional (default=1.0)
   :param degree: Kernel parameter for the poly kernel. See `Kernels in GenSVM
                  <gensvm_kernels_>`_ for the exact implementation of the kernels.
   :type degree: float, optional (default=2.0)
   :param kernel_eigen_cutoff: Cutoff point for the reduced eigendecomposition used with nonlinear
                               GenSVM.  Eigenvectors for which the ratio between their corresponding
                               eigenvalue and the largest eigenvalue is smaller than the cutoff will
                               be dropped.
   :type kernel_eigen_cutoff: float, optional (default=1e-8)
   :param verbose: Enable verbose output
   :type verbose: int, (default=0)
   :param random_state: The seed for the random number generation used for initialization where
                        necessary. See the documentation of
                        ``sklearn.utils.check_random_state`` for more info.
   :type random_state: None, int, instance of RandomState
   :param max_iter: The maximum number of iterations to be run.
   :type max_iter: int, (default=1e8)
   
   .. attribute:: coef_
   
      *array, shape = [n_features, n_classes-1]* -- Weights assigned to the features (coefficients in the primal problem)
   
   .. attribute:: intercept_
   
      *array, shape = [n_classes-1]* -- Constants in the decision function
   
   .. attribute:: combined_coef_
   
      *array, shape = [n_features+1, n_classes-1]* -- Combined weights matrix for the seed_V parameter to the fit method
   
   .. attribute:: n_iter_
   
      *int* -- The number of iterations that were run during training.
   
   .. attribute:: n_support_
   
      *int* -- The number of support vectors that were found
   
   .. attribute:: SVs_
   
      *array, shape = [n_observations, ]* -- Index vector that marks the support vectors (1 = SV, 0 = no SV)
   
   .. seealso::
   
      :class:`.GenSVMGridSearchCV`
          Helper class to run an efficient grid search for GenSVM.
   
   .. _gensvm_kernels:
       https://gensvm.readthedocs.io/en/latest/#kernels-in-gensvm
   
   
   
   .. py:method:: GenSVM.fit(X, y, sample_weight=None, seed_V=None)
      :noindex:
      :module: gensvm.core
   
      Fit the GenSVM model on the given data
      
      The model can be fit with or without a seed matrix (``seed_V``). This
      can be used to provide warm starts for the algorithm.
      
      :param X: The input data. It is expected that only numeric data is given.
      :type X: array, shape = (n_observations, n_features)
      :param y: The label vector, labels can be numbers or strings.
      :type y: array, shape = (n_observations, )
      :param sample_weight: Array of weights that are assigned to individual samples. If not
                            provided, then the weight specification in the constructor is used
                            ('unit' or 'group').
      :type sample_weight: array, shape = (n_observations, )
      :param seed_V: Seed coefficient array to use as a warm start for the optimization.
                     It can for instance be the :attr:`combined_coef_
                     <.GenSVM.combined_coef_>` attribute of a different GenSVM model.
                     This is only supported for the linear kernel.
      
                     NOTE: the size of the seed_V matrix is ``n_features+1`` by
                     ``n_classes - 1``.  The number of columns of ``seed_V`` is leading
                     for the number of classes in the model. For example, if ``y``
                     contains 3 different classes and ``seed_V`` has 3 columns, we
                     assume that there are actually 4 classes in the problem but one
                     class is just represented in this training data. This can be useful
                     for problems were a certain class has only a few samples.
      :type seed_V: array, shape = (n_features+1, n_classes-1), optional
      
      :returns: **self** -- Returns self.
      :rtype: object
      
   
   .. py:method:: GenSVM.predict(X, trainX=None)
      :noindex:
      :module: gensvm.core
   
      Predict the class labels on the given data
      
      :param X: Data for which to predict the labels
      :type X: array, shape = [n_test_samples, n_features]
      :param trainX: Only for nonlinear prediction with kernels: the training data used
                     to train the model.
      :type trainX: array, shape = [n_train_samples, n_features]
      
      :returns: **y_pred** -- Predicted class labels of the data in X.
      :rtype: array, shape = (n_samples, )
      
