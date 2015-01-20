# run like this: . addPath
export C_INCLUDE_PATH=:$HOME/include
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$HOME/sqlite3-3.6.22
export LD_LIBRARY_PATH=:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-5.5/lib64 #where cuda is stored since update
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aiguru2b/leveldb-1.15.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aiguru2b/glue/lib
export PYTHONPATH=:$HOME/lib/python2.7/site-packages
export PYTHONPATH=$PYTHONPATH:$HOME/lib/python2.6/site-packages
#export PYTHONPATH=$PYTHONPATH:/usr/lib64/python2.6/site-packages
export PYTHONPATH=$PYTHONPATH:$HOME/caffe/python
export PYTHONPATH=$PYTHONPATH:$HOME/Imaging-1.1.7
export PATH=/home/aiguru2b/bin:$PATH #make 2.7 the first python on path





