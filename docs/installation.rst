Installation
============

Linux
-----

.. code-block:: bash
   git clone https://github.com/djanloo/quilt.git
   git submodule init
   git submodule update
   sudo apt-install libbost-all-dev
   pip install - r requirements.txt
   make

The code will be built inplace by default and will not be installed globally. 
To make the module accessible be sure to add the path of the root folder to your ``PYTHONPATH``:

.. code-block:: bash
   
   export PYTHONPATH=path/to/quilt:$PYTHONPATH

Windows
-------

Install `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_ and follow instructions for Linux.


MacOs
-----

TODO




The core of quilt is written in C++. The ``main()`` function is contained in quilt/src_cpp/test_file.cpp. To compile it run

.. code-block:: bash
   
   make quilt.exe


