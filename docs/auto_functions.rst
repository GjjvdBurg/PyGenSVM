
.. py:function:: load_grid_tiny()
   :noindex:
   :module: gensvm.gridsearch

   Load a tiny parameter grid for the GenSVM grid search
   
   This function returns a parameter grid to use in the GenSVM grid search.
   This grid was obtained by analyzing the experiments done for the GenSVM
   paper and selecting the configurations that achieve accuracy within the
   95th percentile on over 90% of the datasets. It is a good start for a
   parameter search with a reasonably high chance of achieving good
   performance on most datasets.
   
   Note that this grid is only tested to work well in combination with the
   linear kernel.
   
   :returns: **pg** -- List of 10 parameter configurations that are likely to perform
             reasonably well.
   :rtype: list
   

.. py:function:: load_grid_small()
   :noindex:
   :module: gensvm.gridsearch

   Load a small parameter grid for GenSVM
   
   This function loads a default parameter grid to use for the #' GenSVM
   gridsearch. It contains all possible combinations of the following #'
   parameter sets::
   
       pg = {
           'p': [1.0, 1.5, 2.0],
           'lmd': [1e-8, 1e-6, 1e-4, 1e-2, 1],
           'kappa': [-0.9, 0.5, 5.0],
           'weights': ['unit', 'group'],
       }
   
   :returns: **pg** -- Mapping from parameters to lists of values for those parameters. To be
             used as input for the :class:`.GenSVMGridSearchCV` class.
   :rtype: dict
   

.. py:function:: load_grid_full()
   :noindex:
   :module: gensvm.gridsearch

   Load the full parameter grid for GenSVM
   
   This is the parameter grid used in the GenSVM paper to run the grid search
   experiments. It uses a large grid for the ``lmd`` regularization parameter
   and converges with a stopping criterion of ``1e-8``. This is a relatively
   small stopping criterion and in practice good classification results can be
   obtained by using a larger stopping criterion.
   
   The function returns the following grid::
   
       pg = {
               'lmd': [pow(2, x) for x in range(-18, 19, 2)],
               'kappa': [-0.9, 0.5, 5.0],
               'p': [1.0, 1.5, 2.0],
               'weights': ['unit', 'group'],
               'epsilon': [1e-8],
               'kernel': ['linear']
            }
   
   :returns: **pg** -- Mapping from parameters to lists of values for those parameters. To be
             used as input for the :class:`.GenSVMGridSearchCV` class.
   :rtype: dict
   
