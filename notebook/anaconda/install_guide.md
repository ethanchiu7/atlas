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
    
    wget http://repo.continuum.io/archive/Anaconda2-4.3.0-Linux-x86_64.sh