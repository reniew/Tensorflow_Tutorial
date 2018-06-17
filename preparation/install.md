### Tensroflow 설치하기

이때까지 공부할때는 tensorflow를 사용하지 않고 아나콘다 배포판의 라이브러리들로만 했었는데, 미리미리 tensorflow를 손에 익혀 두는 것이 좋을 것같아 tensorflow tutorial를 따라가며 손에 익혀 보기러 한다.


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

* [Tensroflow 설치하기](#tensroflow-설치하기)
	* [1. Tensorflow](#1-tensorflow)
	* [2. 설치하기](#2-설치하기)
	* [3. 설치 확인하기](#3-설치-확인하기)

<!-- /code_chunk_output -->


#### 1. Tensorflow

tensorflow의 설치방법에는 여러가지가 있지만(참고:[Tensorflow Site](https://www.tensorflow.org/install/)), 여러 방법 중 나는 Ananconda를 통해 다운 받는 방법을 선택했다.

설치하기 전에 tensorflow는 CPU-only 와 GPU도 사용하는 방법 두가지로 나눠져있다.
Tensorflow site의 설명을 확인하면,

> * **TensorFlow with CPU support only**. If your system does not have a NVIDIA® GPU, you must install this version. Note that this version of TensorFlow is typically much easier to install (typically, in 5 or 10 minutes), so even if you have an NVIDIA GPU, we recommend installing this version first. Prebuilt binaries will use AVX instructions.
> * **TensorFlow with GPU support**. TensorFlow programs typically run significantly faster on a GPU than on a CPU. Therefore, if your system has a NVIDIA® GPU meeting the prerequisites shown below and you need to run performance-critical applications, you should ultimately install this version.

그리고 GPU-version을 사용하기 위해서는 요구사항들이 있다.
* CUDA® Toolkit 9.0
* The NVIDIA drivers associated with CUDA Toolkit 9.0
* cuDNN v7.0
* GPU card with CUDA Compute Capability 3.0 or higher for building from source and 3.5 or higher for our binaries

요구사항에 대한 자세한 내용들을 사이트를 참고하자([Tensorflow](https://www.tensorflow.org/install/install_windows), [NVIDIA](https://developer.nvidia.com/))

현재 나는 따로 GPU가 없기 떄문에 일반 버전으로 설치하기로 했다.


#### 2. 설치하기

1. 우선 관리자 권한으로 Anaconda Prompt를 실행시켜 준다.
(Anaconda를 설치하지 않은 사람은 Anaconda를 먼저 설치한다.([아나콘다 설치](https://www.anaconda.com/download/)))

2. 이후 pip 버전을 최신버전으로 맞춰준다.
```python
C:> python -m pip install --upgrade pip
```
3. tensorflow이름으로 conda환경만들기
```python
C:> conda create -n tensorflow pip python=3.5
```
4. conda 환경 활성화
```python
C:> activate tensorflow
```
활성화를 하게 되면 prompt가 (tensorflow) 로 바뀌게 된다.
```python
(tensorflow)C:>
```

5. CPU-only 또는 GPU 두 가지 버전 중 하나를 골라서 설치한다.

* CPU-only
```python
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow
```
* GPU version
```python
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow-gpu
```

여기까지 별 다른 error가 뜨지 않았다면 잘 설치 된 것이다.
그러면 이제 제대로 설치가 되었는지 확인 해 보자.

#### 3. 설치 확인하기

Anaconda를 통해 설치했기 떄문에 anaconda prompt를 통해 실행 해본다

1. Anaconda 실행 후 Python 실행
```python
(tensorflow)C:> python
```
2. tensorflow import 후 사용
```Python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```
3. 만약 잘 설치되었다면, 다음과 같은 문장이 출력 될 것이다.
```python
Hello, TensorFlow!
```

다음 부터는 이제 TensorFlow의 공식 Tutorial([TensorFlow Tutorial](https://www.tensorflow.org/tutorials/)) 문서를 보며 하나씩 따라 해보도록 하겠다.
