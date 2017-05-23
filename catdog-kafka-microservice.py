import tensorflow as tf
import xgboost
import pickle
from kafka import KafkaConsumer, KafkaProducer
from inception_v3 import inception_v3
from inception_v3 import inception_v3_arg_scope
import os
import inception_preprocessing
import numpy as np
import json
import base64

with open('xgboost_model.p', 'rb') as handle:
    classifier = pickle.load(handle)
    
def predict_cat_dog(probs):
    cat_dog_prob = classifier.predict_proba(np.array(probs).reshape((1,-1)))[0]
    return 'Probabilities: cat {:.1%} dog {:.1%}'.format(cat_dog_prob[0], cat_dog_prob[1])

# DOWNLOAD DATASET 
from urllib.request import urlretrieve
import os
from tqdm import tqdm
import tarfile

inceptionv3_archive = os.path.join('model', 'inception_v3_2016_08_28.tar.gz')

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not os.path.isdir('model'):
    # create directory to store model
    os.mkdir('model')
    # download the model
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='InceptionV3') as pbar:
        urlretrieve(
            # I hope this url stays there
            'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
            inceptionv3_archive,
            pbar.hook)

    with tarfile.open(inceptionv3_archive) as tar:
        tar.extractall('model')
        tar.close()

INCEPTION_OUTPUT_SIZE = 1001
IMAGE_SIZE = 299
CHANNELS = 3 # Red, Green, Blue
INCEPTION_MODEL_FILE = os.path.join('model','inception_v3.ckpt')

slim = tf.contrib.slim

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
      
image_raw = tf.placeholder(dtype=tf.string)
image_data = tf.image.decode_png(image_raw, channels=3)
image = inception_preprocessing.preprocess_image(
            image_data, IMAGE_SIZE, IMAGE_SIZE, is_training=False)

expanded_image  = tf.expand_dims(image, 0)
with slim.arg_scope(inception_v3_arg_scope()):
        logits, _ = inception_v3(expanded_image, num_classes=INCEPTION_OUTPUT_SIZE, is_training=False)

probabilities = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(
        INCEPTION_MODEL_FILE, slim.get_model_variables())
    
with tf.Session() as sess:
    init_fn(sess)
    consumer = KafkaConsumer('catdogimage', group_id='group1')
    producer = KafkaProducer(bootstrap_servers='localhost:9092')
    for message in consumer:
        dto = json.loads(message.value.decode()) # Data Transfer Object
        image_data = base64.b64decode(dto['data'])
        np_probabilities = sess.run([probabilities], feed_dict={image_raw:image_data})
        dto['label'] = predict_cat_dog(np_probabilities)
        dto['data'] = None # no need to send image back
        producer.send('catdoglabel', json.dumps(dto).encode())
        print('Prediction made.', dto['label'])
