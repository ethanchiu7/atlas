# alternate channels
    # do this after install conda
    conda config --add channels conda-forge
    # else 
    conda install -c conda-forge xgboost
    
# specify install package version
    conda search Keras
    conda install package=version
    # install by specified verison
    conda install numpy=1.9.3
    # or update
    conda update numpy=1.93

# install in Linux 

    # find the lastest anaconda package
    https://repo.continuum.io/archive/
    
    # py3 : Anaconda3-2018.12-Linux-x86_64.sh
    wget http://repo.continuum.io/archive/Anaconda2-4.3.0-Linux-x86_64.sh
    
# GLIBC

    [/home/iflow/anaconda3] >>> /home/iflow/tuixing.zx/env/anaconda3
    PREFIX=/home/iflow/tuixing.zx/env/anaconda3
    installing: python-3.7.1-h0371630_7 ...
    tar: Read 8704 bytes from -
    /home/iflow/tuixing.zx/env/anaconda3/pkgs/python-3.7.1-h0371630_7/bin/python: /lib64/libc.so.6: version `GLIBC_2.10' not found (required by /home/iflow/tuixing.zx/env/anaconda3/pkgs/python-3.7.1-h0371630_7/bin/python)
    /home/iflow/tuixing.zx/env/anaconda3/pkgs/python-3.7.1-h0371630_7/bin/python: /lib64/libc.so.6: version `GLIBC_2.7' not found (required by /home/iflow/tuixing.zx/env/anaconda3/pkgs/python-3.7.1-h0371630_7/bin/python)
    /home/iflow/tuixing.zx/env/anaconda3/pkgs/python-3.7.1-h0371630_7/bin/python: /lib64/libc.so.6: version `GLIBC_2.9' not found (required by /home/iflow/tuixing.zx/env/anaconda3/pkgs/python-3.7.1-h0371630_7/bin/python)
    /home/iflow/tuixing.zx/env/anaconda3/pkgs/python-3.7.1-h0371630_7/bin/python: /lib64/libc.so.6: version `GLIBC_2.6' not found (required by /home/iflow/tuixing.zx/env/anaconda3/pkgs/python-3.7.1-h0371630_7/bin/python)
    
- 解决办法