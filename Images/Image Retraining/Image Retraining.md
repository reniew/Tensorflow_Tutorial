### Image Retraining

#### How to Retrain an Image Classifier for New Categories
(이 문서는 Tensorflow의 공식 tutorial 가이드를 따라한 것입니다. ([Tensorflow tutorial](https://www.tensorflow.org/tutorials/image_retraining))

현대의 image recognition model들은 수백만개의 파라미터들을 가지고 있다. 막대한 양의 파라미터들을 처음부터 그냥 학습시키는것은 엄청난 양의 computer power를 필요로 한다. *Transer Learning* 은 우리의 task와 연관된 pre-trained 모델들을 사용함으로써 계산량을 줄이는 손쉬운 방법이다.
이번 튜토리얼에서는 ImageNet으로 학습된 강력한 Image Classifier를 통해 feature extraction을 재사용한 후 상단의 classification layer를 학습해보도록 할 것이다. 더 많은 정보는 [DeCAF : A Deep Convolutional Activation Feature for Generic Visual Recognition](https://arxiv.org/abs/1310.1531)을 통해 보면 된다.

전체 모델을 새롭게 학습하는 것보다는 좋지는 않지만, 이러한 방법은 놀랍게도 많은 Applications, Works(적당한 양의, 수천개의 Trainning data)에서 효과적이고, 이것은 GPU없이 노트북에서도 삼십분정도면 실행시킬 수 있다.
이번 tutorial에서 각자 자신의 image를 가지고 예제를 실행하는 방법에 대해서 알아볼 것이며, 학습과정을 제어할 수 있는 몇 가지 옵션에 대해서 알아볼 것이다.

> NOTE 이번 튜토리얼은 [Codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)에서도 실행 가능하다.

이번 tutorial에서 흔히 *Module* 이라 불리는 사전 학습된 모델들을 사용할 것이다. 처음으로 우리는 [Image feature extraction module](https://www.tensorflow.org/modules/google/imagenet/inception_v3/feature_vector/1) 모델을 사용한다. 이 모델을 inception V3 아키텍쳐를 가지며, ImageNet을 사전 학습했다. 그 후에는 더 많은 옵션([NASNet](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html)/PNASNet, [Mobile NET V1](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html), V2)들을 학습할 것이다.

시작하기 전에, *tensorflow-hub* PIP package를 설치해야 한다. 설치에 대한 자세한 사항은 [Installation instructions](https://www.tensorflow.org/hub/installation)참고

##### Trainning On Flowers

![daisies](https://i.imgur.com/SFFNC0k.jpg)[*Image by Kelly Sikkema*](https://www.flickr.com/photos/95072945@N05/9922116524/)

학습을 시작하기 전 우리는 우리가 인식하고 싶어하는 새로운 클래스의 사진들이 필요할 것이다. 이후에 각자 자신들의 사진으로 학습하는 방법을 배워보고, 그전에 먼저 CC LICENSE인 Flower 사진들을 가지고 학습을 해보도록 하겠다. 사진을 가져오기 위해 아래 명령어를 실행하자.

```
cd ~
curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
// 윈도우의 경우에는 직접 압축을 풀자
```

사진을 받았다면, 우리가 필요한 code를 github에서 clone 하자.

```
mkdir ~/example_code
cd ~/example_code
curl -LO https://github.com/tensorflow/hub/raw/r0.1/examples/image_retraining/retrain.py
```

이후 retrainner 파일을 실행하자, 컴퓨터마다 다르겟지만 30분정도 소요된다.

```
python retrain.py --image_dir ~/flower_photos
```
뿐만 아니라 수많은 옵션들을 확인해보자,

```
python retrain.py -h
```

위의 retrain.py 파일은 방금 다운받은 Flower사진들로 network을 재학습시킬 것이다. 참고로 위의 Flower 사진들은 pre-trained시킨 data인 ImageNet에는 전혀 들어가 있지 않은 사진들이다.

##### Bottlenecks

위의 프로그램을 실행하면, 가장 먼저 사진들에 대해 분석하고, 계산해서 저장하는 값들은 bottleneck값이 된다. 'Bottleneck'이란 실제 분류가 진행되는 마지막 층(Final output layer) 이전의 layer를 지칭하는 비공식적인 용어이다.(*Tensorflow Hub은 이를 "image feature vector"라 부른다*) 위의 뒤에서 두번째 레이어는 인식하려하는 클래스들을 충분히 잘 구별할 수 있게 output값을 내놓도록 학습된다. 즉, 매운 작은 값들을 선택해 잘 분류 하기 위한 충분한 정보들을 가지기 때문에, 이 layer들은 이미지의 의미있는 압축된 요약본이라 할 수 있다. 마지막 레이어를 재학습시키는 것이 잘 작동하는 이유는 1000개의 클래스가 있는 ImageNet을 학습시킨 것이 다른 새로운 종류의 개체를 구별하는데도 유용하기 때문이다.

모든 이미지들은 각각의 Bottleneck을 학습하고 계산하는데 많은 양의 시간을 소모하기 떄문에, 이 속도는 이러한 bottleneck값을 cache화 시켜서 다시 계산될 필요가 없도록 하는 것에 달려있다. */tmp/bottleneck* 에 저장된 default값에 의해 만약 다시 프로그램을 돌린다면, 그들은 저장된 값을 사용함으로써 Bottleneck Part를 기다릴 필요가 없다.  


##### Training

일단 Bottleneck이 완료되면, 네트워크 상단 layer의 실제 학습은 진행된다. 여러 스탭의 결과들을 볼 수 있는데, 각각 'training accuracy', 'validation accuracy', 'cross entropy'를 출력한다. 각각의 값을 살펴보면.
* Training accuracy : 현재 trainning batch 중 제대로 예측한 data의 비율이다.
* validation accuracy : 각각 다른 set으로 부터 무작위로 선택된 이미지의 그룹들의 정확도이다.
  * training accuracy를 핵심 지표로 활용할 경우 noise까지도 학습하게 되서 Overfit될 가능성이 있다. 따라서 실제 성능의 척도는 training data에 포함되지 않은 data들에 대한 성능을 확인해야 한다(= validation accuracy).
  * Traindata에 대한 accuracy는 높은 반면, Validation accuracy는 낮게 나온다면, Overfitting 됬다는 뜻이며, 이는 좋지 않는 결과이다.
* Cross Entropy : loss function으로 학습의 진척도를 보여준다.
 * 우리 학습의 목표는 cross entropy를 가능한 가장 작게 만드는 것이다.
여기에서 우리는 4,000Step(Default)을 학습한다. 각각의 스텝에서 trainning set에서 10개의 무작위 image가 선택한 후 Cache로 부터 각각의 bottleneck을 찾은 후 예측을 위해 마지막 레이어에 통과시킨다. 이러한 예측과 실제 label을 비교해 마지막 레이어의 weight들을 back-propagation 과정을 통해 update시킨다. process가 진행될 수록 우리는 accuracy가 올라가는 것을 확인할 수 있다. 모든 학습이 끝난 후 test를 위해 따로 구분해놓은 set을 예측함으로써 final test accuracy가 나온다. 이 수치는 모델이 얼마나 잘 학습했는지를 알려주는 가장 좋은 지표이다. 우리는 90~95%정도의 수치를 보는데 각각 다른이유는 각각 스탭마다 무작위로 사진을 뽑기 떄문이다.

##### Visualizing the retraining with TensorBoard

`retrain.py` 파일은 tensorboard를 포함한다. tensorboard란 weights, accuracy등의 통계수치, 그래프들을 시각적으로 보여줘서 이해, debug, 최적화에 용이하게 한다.

tensorboard를 실행하기 위해 아래의 명령어를 입력하자

```
tensorboard --logdir /tmp/retrain_logs
```

Tensorboard가 실행되면,web browser에서 [localhost:6006](http://localhost:6006)에 접속하면 된다.

`retrain.py`은 기본적으로 log를 /tmp/retrain_logs에 저장한다. 저장경로는 `--summaries_dir` flag로 바꿀 수 있다.
[TensorBoard's Github Repository](https://github.com/tensorflow/tensorboard)에서 TensorBoard에 대한 더욱 많은 설명과 tip들을 확인할 수 있다.

##### Using Retrained Model
