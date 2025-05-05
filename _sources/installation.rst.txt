Installation
============

Linux
-----
Requires boost at least v1.74:

.. code-block:: bash
   
   sudo apt-install libboost-all-dev

To install quilt run:

.. code-block:: bash
   
   git clone https://github.com/djanloo/quilt.git
   cd quilt
   git submodule init && git submodule update
   pip install .

Windows
-------

Install `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_ and follow instructions for Linux.


MacOs
-----

TODO


The core of the simulator is written in C++. To produce an executable using the ``main()`` function that is contained in quilt/core/main.cpp, run:

.. code-block:: bash
   
   make exe


