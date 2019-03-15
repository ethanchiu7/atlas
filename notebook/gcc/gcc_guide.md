# Local build without root

    https://pachterlab.github.io/kallisto/local_build.html
    
    
# install without sudo

    # try to install glibc 2.14 in a specific path and force it with an env variable

    # https://stackoverflow.com/questions/48591455/glibc-2-14-not-found-what-should-i-do-if-i-am-not-root
    # https://stackoverflow.com/questions/35616650/how-to-upgrade-glibc-from-version-2-12-to-2-14-on-centos
    
# install bison
    # install gnu bison
    mkdir /home/username/gnu-install
    cd /home/username/gnu
    wget https://ftp.gnu.org/gnu/bison/bison-3.3.2.tar.gz
    tar -zxvf bison-3.3.2.tar.gz
    cd bison-3.2
    ./configure --prefix=/home/username/bison-3.3.2
    make
    make install
    export PATH="/home/username/bison-3.3.2/bin:$PATH"
    
# install the GLIBC
    cd /home/username/gnu
    wget https://ftp.gnu.org/gnu/glibc/glibc-2.29.tar.gz
    wget https://ftp.gnu.org/gnu/glibc/glibc-2.24.tar.gz
    wget https://ftp.gnu.org/gnu/glibc/glibc-2.18.tar.gz
    
    tar -zxvf glibc-2.18.tar.gz
    mkdir /home/datamining/tuixing.zx/env/glibc-2.18-build
    cd glibc-2.18
    
    # add to PATH
    export LD_LIBRARY_PATH=/xx/gcc/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/xx/gcc/lib64:$LD_LIBRARY_PATH
    
    
# 解决too old: as ld

    wget ftp://ftp.gnu.org/gnu/autoconf/autoconf-2.68.tar.gz
    tar zxvf autoconf-2.68.tar.gz
    cd autoconf-2.68
    ./configure --prefix=/usr/
    make && make install

    
# tensorflow GLIBC

    https://zhuanlan.zhihu.com/p/33059558