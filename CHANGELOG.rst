Change Log
----------

Version 0.2.3
^^^^^^^^^^^^^

- Bugfix for prediction with gamma = 'auto'

Version 0.2.2
^^^^^^^^^^^^^

- Add error when unsupported ShuffleSplits are used

Version 0.2.1
^^^^^^^^^^^^^

- Update docs
- Speed up unit tests

Version 0.2.0
^^^^^^^^^^^^^

- Add support for interrupting training and retreiving partial results
- Allow specification of sample weights in GenSVM.fit()
- Extract per-split durations from grid search results
- Add pre-defined parameter grids 'tiny', 'small', and 'full'
- Add code for prediction with kernels
- Add unit tests
- Change default coef in poly kernel to 1.0 for inhomogeneous kernel
- Minor bugfixes, documentation improvement, and code cleanup
- Add continuous integration through Travis-CI.

Version 0.1.6
^^^^^^^^^^^^^

- Fix segfault caused by error function in C library.
- Add "load_default_grid" function to gensvm.gridsearch
