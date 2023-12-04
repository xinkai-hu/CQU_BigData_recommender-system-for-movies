# 警告

***作者在配置环境时曾经历了千难万险，在此奉劝读者务必在头脑清醒时小心安装。代码环境对各项配置的要求非常严格，环环相扣，安装版本偏差很容易产生意想不到的问题。***

***本指导中与案例正文不符的部分以案例正文为准。若配置环境过程中出现本指导和案例正文中均未提到的问题，请根据报错信息修改。***

作者经历过的各类问题：
- 硬件配置不满足 CUDA 和 NCCL 的要求（NCCL 不支持 WSL）
- CUDA 版本与 PyTorch 版本不匹配
- PyTorch 版本过低导致 PyG 的某些依赖不能正常运行
- Python 版本过高或过低（PyG 要求 Python 版本在 3.8 至 3.11 之间）
- Java 版本不足以运行 Neo4j 图数据库（要求 jdk-17 及以上）
- g++ 版本过高或过低（horovod 要求 g++-5 及以上；实测 g++-10 及以上可能产生问题）
- CMake 版本过低（horovod 要求 CMake 3.13 及以上）
- OpenMPI 版本（horovod 指定 OpenMPI 4.0.0）
- 安装 horovod 时尽量参考官方 GitHub 仓库（而不是官方网站，因为官方网站的更新不如 GitHub 更新及时）

另外，在代码运行过程中遇到过的问题：
- horovod.torch 运行应使用 horovodrun 命令
- 但是 horovod.spark.torch 运行时应使用 python 命令

参考链接：
- [GPU加速型实例自动安装GPU驱动](https://support.huaweicloud.com/usermanual-ecs/ecs_03_0199.html)
- [NCCL 安装](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#debian)
- [Horovod 安装](https://github.com/horovod/horovod#install)
- [Horovod on GPU 安装](https://github.com/horovod/horovod/blob/master/docs/gpus.rst#horovod-on-gpu)
- [Horovod with MPI 安装](https://github.com/horovod/horovod/blob/master/docs/mpi.rst#horovod-with-mpi)


# 配置总览

- 华东-上海一
- x86计算
- GPU加速型pi2
- Ubuntu 20.04 server 64bit for GPU(40GiB)
- Driver-450.203.03(Tesla)
- CUDA-11.0.3
- CUDNN-8.1.1
- g++ 9
- CMake 2.22
- NCCL 2.18.5
- OpenMPI 4.0.0
- Python 3.8.10
- Java 17.0.9
- Hadoop 3.3.6
- Spark 3.5.0
- pip 23.3.1
- PyTorch 1.9.1
- horovod 0.28.1

# 华为云服务器

区域：华东-上海一

CPU架构：x86计算

实例类型：GPU加速型pi2

镜像：Ubuntu 20.04 server 64bit for GPU(40GiB)

自动安装驱动：Driver-450.203.03(Tesla) CUDA-11.0.3 CUDNN-8.1.1

# 创建用户

添加用户
```sh
useradd -m hadoop -s /bin/bash
passwd hadoop
```

权限
```sh
visudo
```

切换用户
```sh
su hadoop
```

配置密钥
```sh
ssh localhost
cd ~/.ssh
ssh-keygen -t rsa
cat id_rsa.pub >> authorized_keys
chmod 600 authorized_keys
```

# 更新

```sh
sudo apt update
sudo apt upgrade
sudo apt-get -y update
```

# GCC

可选（默认已安装）
```sh
sudo apt-get install gcc-9 g++-9 make
```

# CMake

```sh
sudo apt install cmake
```

# CUDA

查询驱动是否安装成功
```sh
$ nvidia-smi
```

检查CUDA版本是否正确
```sh
$ /usr/local/cuda/bin/nvcc -V
```

检查CUDA是否正常
```sh
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

环境变量
```sh
export CUDA_HOME=$PATH:/usr/local/cuda
export CPATH=$CPATH:$CUDA_HOME/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export PATH=$PATH:$CUDA_HOME/bin
```

若异常：根据[GPU加速型实例自动安装GPU驱动（Linux）](https://support.huaweicloud.com/usermanual-ecs/ecs_03_0199.html)选择合适的链接
```sh
wget -t 10 --timeout=10 https://hgcs-drivers-cn-east-3.obs.cn-east-3.myhuaweicloud.com/release/script/auto_install.sh && bash auto_install.sh
sudo bash auto_install.sh
```

# NCCL 2

下载
```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install libnccl2=2.18.5-1+cuda11.0 libnccl-dev=2.18.5-1+cuda11.0
```

环境变量
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include/nccl.h
```

# OpenMPI

下载
```sh
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
```

安装
```sh
mkdir build
cd build
../configure --prefix=/usr/local/openmpi
sudo make all install
```

环境变量
```sh
export PATH=$PATH:/usr/local/openmpi/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib
export CPATH=$CPATH:/usr/local/openmpi/include
```

查看MPI环境是否正常
```sh
which mpirun
```

# Java

下载
```sh
wget https://download.oracle.com/java/17/archive/jdk-17.0.9_linux-x64_bin.tar.gz
```

环境变量
```sh
export JAVA_HOME=/usr/local/java
export PATH=$PATH:$JAVA_HOME/bin
```

# Hadoop

下载
```sh
wget https://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/stable/hadoop-3.3.6.tar.gz
```

环境变量
```sh
export HADOOP_HOME=/usr/local/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
```

权限
```sh
sudo chown -R hadoop:hadoop /usr/local/hadoop
```

# Spark

下载
```sh
wget https://mirrors.tuna.tsinghua.edu.cn/apache/spark/spark-3.5.0/spark-3.5.0-bin-without-hadoop.tgz
```

环境变量
```sh
export SPARK_HOME=/usr/local/spark
export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip
export PYSPARK_PYTHON=python3
export PATH=$PATH:$SPARK_HOME/bin
```

权限
```sh
sudo chown -R hadoop:hadoop /usr/local/spark
```

配置
```sh
cd /usr/local/spark
cp conf/spark-env.sh.template conf/spark-env.sh
vim conf/spark-env.sh
```
添加
```sh
export SPARK_DIST_CLASSPATH=$(/usr/local/hadoop/bin/hadoop classpath)
```

测试
```sh
/usr/local/spark/bin/run-example SparkPi
```

# Neo4j

下载
```sh
wget https://dist.neo4j.org/neo4j-community-5.12.0-unix.tar.gz
```

环境变量
```sh
export PATH=$PATH:/usr/local/neo4j/bin
```

远程访问
```sh
vim /usr/local/neo4j/conf/neo4j.conf
```
修改
```sh
dbms.connectors.default_listen_address=0.0.0.0
```

防火墙：端口7474、7687

安装 [Neo4j Connector for Apache Spark](https://neo4j.com/product/connectors/apache-spark-connector/)（需注册后下载，运行时通过 `--jars` 参数传入 Spark）

# pip

更新
```
pip install --upgrade pip
```

# PyTorch

下载
```sh
pip install torch==1.9.1+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

# PyG

下载
```sh
pip install torch_geometric -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# TorchScatter

下载
```sh
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.1+cu110.html
```

# TensorBoard

下载
```
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# Horovod

安装
```sh
HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod[spark,pytorch] -i https://pypi.tuna.tsinghua.edu.cn/simple
```

检查
```
horovodrun --check-build
```


# 测试代码

PyTorch on Spark with horovod: [horovod/examples/spark/pytorch/pytorch_spark_mnist.py at master · horovod/horovod (github.com)](https://github.com/horovod/horovod/blob/master/examples/spark/pytorch/pytorch_spark_mnist.py)
运行
```
python3 <file_path>.py --master=local --num-proc=1
```

PyTorch with horovod: [horovod/examples/pytorch/pytorch_mnist.py at master · horovod/horovod (github.com)](https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py)
运行
```
horovodrun -np 1 python3 <file_path>.py
```
或者
```
python3 <file_path>.py --num-proc=1 --hosts="localhost:1" --communication="mpi"
```
