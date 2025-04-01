import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from caption_processing import *
from feature_loading import *
from encoder import Encoder
from decoder import Decoder


MAX_INPUT_SIZE = 60000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EPOCH_NUM = 21
UNITS = 512
ENC_DIM = 256
FEATURES_V3_SHAPE = 2048
ATTENTION_V3_SHAPE = 64

def save_model(encoder, decoder, save_path="./saved_model"):
    # Сохраняем encoder и decoder в формате SavedModel
    tf.saved_model.save(encoder, f"{save_path}/encoder")
    tf.saved_model.save(decoder, f"{save_path}/decoder")
    print(f"Model saved to {save_path}")

def load_model(save_path="./saved_model"):
    encoder = tf.saved_model.load(f"{save_path}/encoder")
    decoder = tf.saved_model.load(f"{save_path}/decoder")
    print(f"Model loaded from {save_path}")
    return encoder, decoder

def __feature_caption_mapper(img_path, caption):
    img_features = np.load(img_path.decode('utf-8') + '.npy')
    return img_features, caption


def generate_train_dataset(img_paths, captions,
                           batch_size=BATCH_SIZE,
                           buffer_size=BUFFER_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, captions))
    dataset = dataset.map(lambda img, cap:
        tf.numpy_function(__feature_caption_mapper, [img, cap],
                          [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset.shuffle(buffer_size).batch(batch_size)


def calculate_loss(expected, predicted, loss_obj):
    mask = tf.math.logical_not(tf.math.equal(expected, 0))
    loss_ = loss_obj(expected, predicted)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


captions, img_paths = load_captions()
captions = captions[:MAX_INPUT_SIZE]
img_paths = img_paths[:MAX_INPUT_SIZE]

token_seqs, tokenizer = captions_to_token_seqs(captions)
token_len = len(token_seqs[0])
vocab_size = len(tokenizer.word_index) + 1

unique_paths = sorted(set(img_paths))
v3_features_model = get_features_model()
fetch_image_features(unique_paths, v3_features_model)

train_img_paths, test_img_paths, train_img_caps, test_img_caps = train_test_split(img_paths, token_seqs, test_size=0.2, random_state=0)
print('Train captions: %d; Train images: %d' %
      (len(train_img_caps), len(set(train_img_paths))))

train_dataset = generate_train_dataset(train_img_paths, train_img_caps)

batch_num = len(train_img_paths) // BATCH_SIZE

encoder = Encoder(ENC_DIM)
decoder = Decoder(ENC_DIM, UNITS, vocab_size)
optimizer = tf.keras.optimizers.Adam()

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def __load_checkpoints(checkpoint_path="./checkpoints/train"):
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder,
                                     optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_path, max_to_keep=5)

    start_epoch = 1
    latest = checkpoint_manager.latest_checkpoint
    if latest:
        checkpoint.restore(latest)
        start_epoch = int(latest.split('-')[-1]) + 1
    return start_epoch, checkpoint_manager


@tf.function
def train_step(features, expected):
    # Flush decoder state as new image captions are not related to old ones
    hidden = decoder.reset_state(batch_size=expected.shape[0])
    # Start training with start token
    input = tf.expand_dims([tokenizer.word_index[CAP_START]] *
                           expected.shape[0], 1)
    loss = 0
    with tf.GradientTape() as tape:
        enc_features = encoder(features)
        for i in range(1, expected.shape[1]):
            predictions, hidden, _ = decoder(input, enc_features, hidden)
            loss += calculate_loss(expected[:, i], predictions, loss_obj)
            # Teacher forcing
            input = tf.expand_dims(expected[:, i], 1)
    total_loss = (loss / int(expected.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables


    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return total_loss


def __train_model(train_ds, epoch_num):
    start_epoch, checkpoint_manager = __load_checkpoints()
    losses = []
    if start_epoch == epoch_num:
        return losses

    for epoch in range(start_epoch, EPOCH_NUM):
        start = time.time()
        total_loss = 0

        for (batch, (features, expected)) in enumerate(train_ds):
            loss = train_step(features, expected)
            total_loss += loss

        # storing the epoch end loss value to plot later
        losses.append(total_loss / batch_num)

        checkpoint_manager.save()

        print('Epoch {}: loss {:.6f}, time {:.2f}'.format(
            epoch, (total_loss / batch_num).numpy(), time.time() - start))

    # encoder.save('encoder_model.h5')
    # decoder.save('decoder_model.h5')
    return losses

    # No training will be processed if all checkpoints are present


losses = __train_model(train_dataset, EPOCH_NUM)
if losses:
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()


def __test_model(image_path):
    attention_plot = np.zeros((token_len, ATTENTION_V3_SHAPE))
    hidden = decoder.reset_state(batch_size=1)
    input = tf.expand_dims(load_preformatted_image(image_path)[0], 0)
    features = v3_features_model(input)
    features = tf.reshape(features,
        (features.shape[0], -1, features.shape[3]))

    features = encoder(features)
    dec_input = tf.expand_dims([tokenizer.word_index[CAP_START]], 0)
    result = []

    for i in range(token_len):
        predictions, hidden, attention_weights = decoder(
            dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == CAP_END:
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def __draw_attention_plot(image_path, result, attention_plot):
    temp_image = np.array(Image.open(image_path))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

custom_images = [
    'https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?ixlib=rb-1.2.1&w=1000&q=80',
    'https://upload.wikimedia.org/wikipedia/commons/6/66/An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg',
    'https://ak7.picdn.net/shutterstock/videos/6095327/thumb/1.jpg',
    'https://zooclub.ru/attach/8055.jpg'
]

cnt = 0
for image_url in custom_images:
    image_path = tf.keras.utils.get_file('image_%d.jpg' % cnt, origin=image_url)
    result, attention_plot = __test_model(image_path)
    print('Prediction: %s' % str(result))
    __draw_attention_plot(image_path, result, attention_plot)
    cnt = cnt + 1


