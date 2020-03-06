Kernels in GenSVM
-----------------

Kernels in GenSVM are implemented as follows.

- Radial Basis Function (RBF):

.. math::

  k(x_1, x_2) = \exp(-\gamma \| x_1 - x_2 \|^2 )

- Polynomial:

.. math::

  k(x_1, x_2) = (\gamma x_1'x_2 + coef)^{degree}

- Sigmoid:

.. math::

  k(x_1, x_2) = \tanh(\gamma x_1'x_2 + coef)
