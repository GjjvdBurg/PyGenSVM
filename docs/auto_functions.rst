
.. py:function:: load_default_grid()
   :noindex:
   :module: gensvm.gridsearch

   Load the default parameter grid for GenSVM
   
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
   
