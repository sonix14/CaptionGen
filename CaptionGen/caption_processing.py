import os
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import shuffle
import tensorflow as tf


# Configuration of paths
TRAIN_CAP_FILE = 'annotations/captions_train2014.json'
TRAIN_IMG_DIR = 'train2014/'
TRAIN_IMG_PREFIX = 'COCO_train2014_'
IMG_SUFFIX_FMT = '%012d.jpg'

# Lexical configuration
CAP_START = '<start>'
CAP_END = '<end>'

# Dimensions, indexes and amounts of data
TOP_FREQ_NUM = 10000 # Max number of most frequent tokens to extract

def load_captions(cap_file=TRAIN_CAP_FILE, img_dir=TRAIN_IMG_DIR,
                  img_prefix=TRAIN_IMG_PREFIX,
                  img_suffix_fmt=IMG_SUFFIX_FMT):

    if not os.path.exists(cap_file):
        raise FileNotFoundError('File with captions not found in path: %s' % cap_file)
    if not os.path.exists(img_dir):
        raise FileNotFoundError('Directory with images not found, path: %s' % img_dir)

    f = open(cap_file, 'r')
    content = json.load(f)
    f.close()

    captions = []
    img_paths = []

    for entry in content['annotations']:
        caption = CAP_START + ' ' + entry['caption'] + ' ' + CAP_END
        img_path = img_dir + img_prefix + img_suffix_fmt % (entry['image_id'])

        captions.append(caption)
        img_paths.append(img_path)

    # Shuffle captions and paths to images all together using Sklearn
    return shuffle(captions, img_paths, random_state=1)
def captions_to_token_seqs(captions, top_freq_num=TOP_FREQ_NUM):
    # Use tokenizer from Keras
    # Extract top_freq_num most frequent words from all captions
    # Generate sequences of tokens which represent captions
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=top_freq_num,
        oov_token='<UNDEF>',
        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(captions)
    # seqs = tokenizer.texts_to_sequences(captions)
    # Now we need to extend length of each list of tokens to the max one
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    seqs = tokenizer.texts_to_sequences(captions)
    return tf.keras.preprocessing.sequence.pad_sequences(
        seqs, padding='post'), tokenizer
