### Image Recognition

#### Image Recognition
(이 문서는 Tensorflow의 공식 tutorial 가이드를 따라한 것입니다. ([Tensorflow tutorial](https://www.tensorflow.org/tutorials/image_recognition))

사람의 뇌는 어떠한 사진을 보고 사자인지, 표범인지 구별하거나, 사람의 얼굴의 인식하는 것을 매우 쉽게 한다. 그러나 이러한 일들은 컴퓨터에게는 쉽지 않은 일이다.
지난 몇년동안 이러한 분야에서 machine learning 은 많은 성과를 이뤘다. [CNN](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/) 모델을 통해 우리는 visual recognition 분야에서 reasonable한 perfomance를 보여줬다.

많은 연구들은 computer vision분야에서 학문적인 기준점이 되는 [ImageNet](http://www.image-net.org/)에 대해 계속해서 발전해나가는 연구들을 선보였다.
이후 많은 연구들이 계속해서 state-of-art한 성과를 보여줬다. : [QuocNet](https://static.googleusercontent.com/media/research.google.com/en//archive/unsupervised_icml2012.pdf), [AlexNet](https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf), [Inception (GoogLeNet)](https://arxiv.org/abs/1409.4842), [BN-Inception-v2](https://arxiv.org/abs/1502.03167)
이후에도 계속해서 Google은 많은 논문들을 작성하고 기존의 모델들을 수정해 새롭게 공개했다. 그 중 우리는 가장 최근 모델인 [Inception-v3](https://arxiv.org/abs/1512.00567)에 대해서 알아보겠다.

Inception-v3 모델은 [ImageNet](http://www.image-net.org/)(2012년 부터 진행된 같은 데이터를 사용하는 Visual Recognition Challenge), Computer vision분야에서 가장 기본적인 task는 전체 이미지를 [1000개의 class](http://image-net.org/challenges/LSVRC/2014/browse-synsets)로 구분하는 문제이다.(ex. "Zebra", "Dalmation", "Dishwasher"),
예를 들면 아래의 결과는 AlexNet이 Classify한 결과이다.
![IR1](https://i.imgur.com/GYPYNQQ.png)

모델들을 비교하기 위해 각 모델들이 top 5 guesses에 대해 얼마나 예측을 실패 한지를 비교하였다.(top-5 error rate)
* [AlexNet](https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) : 15.3%
* [Inception (GoogLeNet)](https://arxiv.org/abs/1409.4842) : 6.67%
* [BN-Inception-v2](https://arxiv.org/abs/1502.03167) : 4.9%
* [Inception-v3](https://arxiv.org/abs/1512.00567) : 3.46%

이번 Tutorial에서는 [Inception-v3](https://arxiv.org/abs/1512.00567) model을 사용하는 방법에 대해서 배울 것이다. Python 또는 C++ 에서 [1000개의 클래스들](http://image-net.org/challenges/LSVRC/2014/browse-synsets)로 분류하는 방법에 대해서 배워 볼 것이다.


##### Usage with Python API

classify_image.py 파일을 [이곳](https://github.com/reniew/tensorflow_tutorials)에서 다운받는다.
코드 전문을 보면,
```Python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  run_inference_on_image(image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  ```
