### MNIST

#### TensorFlow Layers 가이드 : Convoltional Neural Network 만들기
(이 문서는 Tensorflow의 공식 tutorial 가이드를 따라한 것입니다. ([Tensorflow tutorial](https://www.tensorflow.org/tutorials/layers#training_and_evaluating_the_cnn_mnist_classifier)))

**목차**


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

* [MNIST](#mnist)
	* [TensorFlow Layers 가이드 : Convoltional Neural Network 만들기](#tensorflow-layers-가이드-convoltional-neural-network-만들기)
		* [시작하기](#시작하기)
		* [CNN 소개](#cnn-소개)
		* [CNN MNIST 분류기 만들기](#cnn-mnist-분류기-만들기)

<!-- /code_chunk_output -->

[toc]



* TF는 쉽게 Neural network을 블록쌓듯 만들 수 있게 high-level의 API로써 Tensorflow LayerModule을 제공한다.([Module 참고](https://www.tensorflow.org/api_docs/python/tf/layers))
* 이번 Tutorial에서는 손글씨 MNIST데이터를 학습하기 위한 CNN모델을 만들기 위해 Layer를 만들 것이다.
* ![mnist](https://i.imgur.com/klzPlWu.png)
* MNIST 데이터는 위와 같은 28x28-pixel의 0~9까지의 숫자 데이터로 이루어져있으며, trainning data 는 6만개, test data는 만개로 이루어져 있다.

##### 시작하기

아래와 같은 code로 이루어진 cnn_mnist.py파일을 만든다.
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

## Tensorflow는 5가지 ligging type을 제공한다.( DEBUG, INFO, WARN, ERROR, FATAL )
## default값은 WARN
## 우리는 이를 INFO로 바꿔준다.
tf.logging.set_verbosity(tf.logging.INFO)

# Logic들은 이곳에 추가된다.

if __name__ == "__main__":
  tf.app.run()
```
최종 코드는 다음에서 확인 가능하다. ([Final Code](https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/layers/cnn_mnist.py))

##### CNN 소개
CNN은 연속된 필터를 raw pixel 이미지 데이터에 적용한다. 이를 통해 high-level의 특징을 뽑아내고 학습한다. CNN은 세가지 구성요소로 이루어진다.

* **Convolution Layers** 는 특정한 수의 필터를 이미지에 적용시킨다. 각각의 subregion에 합성곱(Convolution)됨으로써 single value를 만든다. 그 이후 일반적으로 ReLU 활성화 함수를 통과시켜 비선형(non-linear)한 output을 만든다.
* **Pooling Layers** 는 [Image data를 downsampling](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer)하는 과정이다. Feature map의 차원을 감소시킴으로써 수행시간을 감소시킨다. 일반적으로 Max-pooling을 사용한다.(최고값만을 남기고 나머지는 버린다.)
* **Dense (fully connected) layers** 는 분류를 수행하는 Layer이다. 여기서는 모든 layer들이 이전 모든 노드들과 연결되어있다.
일반적으로 CNN은 위의 요소들을 블록처럼 쌓으면서 만들어지는 model이다.

##### CNN MNIST 분류기 만들기
다음과 같은 arichitecture를 가지는 모델을 만들어 MNIST 데이터를 분류할 것이다.

1. **Convolutional Layer #1** :32개의 5x5 filter로 구성, ReLU 함수 사용  
2. **Pooling Layer #1** : 2x2 Filter로 2stride를 적용해  max pooling사용  
3. **Convolutional Layer #2**: 64개의 5x5 filter로 구성, ReLU 함수 사용
4. **Pooling Layer #2**: 2와 같은 구조
5. **Dense Layer #1**: 1,024개의 뉴련, 0.4의 dropout 사용

[Tf.layer](https://www.tensorflow.org/api_docs/python/tf/layers) 모듈은 아래의 3가지 type의 layer를 만들 수 있다.

>* **conv2d()**. Constructs a two-dimensional convolutional layer. Takes number of filters, filter kernel size, padding, and activation function as arguments.
>* **max_pooling2d()** . Constructs a two-dimensional pooling layer using the * max-pooling algorithm. Takes pooling filter size and stride as arguments.
>* **dense()**. Constructs a dense layer. Takes number of neurons and activation function as arguments.

이전의 cnn_mnist.py를 열어 다음과 같이 cnn_model_fn 함수를 만든다.

```python
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer 28x28 input size
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1 (32개의 5x5 필터, padding = "same"은 입력과 출력 크기가 같도록 유지시킨다. 활성화함수는 ReLU)
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1 (2x2, stride2로 Max-pooling사용)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2(64개의 5x5필터)
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer (3136x1로 reshape 한뒤 1024x1로 만든후 dropout실행 )
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
```

이제부터 함수 하나씩 살펴보자

1. **tf.layer.conv2d**
```python
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
```
* *Convolutional Layer #1* & *Convolutional Layer #2*
* 모든예시는 첫번째 conv인 Convolutional Layer #1을 예로 한다.
* *input*은 다음과 같은 shape을 가져야 한다.
  * *[batch_size, image_height, image_width, channels]*
  * 여기서는 [batch_size, 28, 28, 1]이 된다.
* filter는 필터의 수를 뜻한다.
* *kernel_size* 는 filter의 크기를 뜻한다.
  * *[height, width]* (여기서는, [5, 5]).
* *padding* 은 *valid* 값(default) 혹은 *same* 값으로 설정한다.
  * *valid* 는 패딩 없음
  * *same* 은 0패딩을 포함시켜, 출력 크기가 입력 크기 와 동일 하게 한다.
* *activation* 은 활성화 함수를 선택한다. 여기서는 ReLU함수를 사용하므로 *tf.nn.relu* 로 하였다.
* 따라서 conv2d의 아웃풋인 conv1의 shape은 *[batch_size, 28, 28, 32]* 을 가진다.

2. **tf.layer.max_pooling2d**
* *Pooling Layer #1* & *Pooling Layer #2*
* 모든 예시는 첫번째 pool인 Pooling Layer #1으로 한다.
* Pooling 필터를 2x2 로 하고 stride를 2로하여서 겹치는 부분없이 최대값을 뽑아내는 max pooling을 사용하였다.
* 풀링을 함으로써 height과 width의 size가 절반으로 줄었다.( *[batch_size, 14, 14, 32]* )

3. **Dense Layer**
* Fully Connected Layer를 만들기 위한 과정이다.
* 먼저 우리는 feature map(pool2)을 tf.reshape 함수로 평평하게 만들어 준다.
```python
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
```
* reshpae()함수에서 -1은 batch_size에 맞게 설정되도록 한다. 따라서 reshape을 하고 나면 shape가 *[batch_size, 3136]* 가 된다.
```python
dropout = tf.layers.dropout(
    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
```
* 평평하게 만든 뒤 tf.layers.dense 함수를 사용해 dense layer와 연결한다.
  * input값은 평평하게 만든 feature map 값(pool2_flat)이다
  * units 값은 dense layer의 뉴런의 수이다. (1024)
  * 활성화 함수는 ReLU 사용
* dense layer와 연결한 뒤 Dropout 사용 (0.4)
  * tf.layers.dropout함수 에서 training 인자는 현재 학습 상태인지 예측하는 상태인지를 Bool 값으로 받는다. 학습상태에서만 dropout 실행
* Output 값은 *[batch_size, 1024]*

4. **Logits Layer**
* 마지막으로 1024개의 뉴런을 10개의 뉴런(0~9 예측 위해) 으로 만드는 Logit layer 만든다.
```python
logits = tf.layers.dense(inputs=dropout, units=10)
```
* Linear activation 사용(default)
* 최종 output은 *[batch_size, 10]*

5. **Generate Predictions**
* "Class" 와 "probabilities"값 리턴
  * Class 는 argmax값 통해 가장 값이 큰 것의 index 리턴
  * Probabilities는 softmax값 리턴(tf.nn.softmax 함수 사용)
* mode 가 예측일 때만 리턴

```python
predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
}
if mode == tf.estimator.ModeKeys.PREDICT:
  return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  ```

6. **Calculate Loss**
* MNIST 분류 문제는 multi-classication 문제이므로 일반적으로 cross entropy 사용
* 아래의 코드는 trainning과 evaluation 모두에서 사용된다.
```python
onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits)
  ```
* *label* one-hot 인코딩 방식이므로 tf.one_hot 함수 사용
  * one-hot 함수의 *indices* 인자는 1로 표시할 index
  * depth는 벡터의 크기로 class수가 된다.

7. **Configure trainning Op**
* 학습하는 동안 Loss를 갱신시킬 방법을 선택한다.
* 이번 예제에서는 SGD(Stochastic Gradient Descent)를 사용한다.(learning_rate = 0.001)

```Python
if mode == tf.estimator.ModeKeys.TRAIN:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
```

8. **Add Evaluation Matrics**
* accuracy를 추가한 Estimator를 만든다.
```python
eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  ```  



##### 실제로 MNIST 분류기를 학습시키고 평가하기


1. **Trainning data와 Test data를 불러온다.**
* main()함수를 만들어서 data를 불러온다.
```python
def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  ```
* data들은 numpy배열로 이루어져 있다.

2. **Create Estimator**
* main()함수에 estimator를 추가한다.
```python
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
```
* model_fn 인자로는 우리가 위에서 만들었던 함수를 넣는다.
* model_dir은 model이 저장될 위치를 지정한다.

3. **Set up Logging Hook**

* logging할 방법을 정한다.
```python
tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
```
* 여기서 [tf.train.SessionRunHook](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook)을 사용해 [tf.train.LoggingTensorHook](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook)만들 수 있다. 이를 통해 softmax값인 확률값을 logging할 수 있다.

4. **모델 학습하기**

* main()함수에 다음의 train함수를 추가한다.
```python
# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])
  ```

5. **모델 평가하기**

* 다음의 함수를 main()에 추가해서 Estimator를 추가한다.
```python
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
 ```

이제 완성된 cnn_mnist.py파일을 실행시킨다.
```
Run cnn_mnist.py.
```
*정확도 : 97.3%*
