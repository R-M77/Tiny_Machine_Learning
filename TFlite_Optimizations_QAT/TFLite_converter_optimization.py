import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfdata

# Normalize data - format images
def format_img(img, label):
    img = tf.image.resize(img, (224,224)) /255.0
    return img, label

# Load dataset 
(train_img, validate_img, test_img), metadata = tfdata.load('cats_vs_dogs', split= ['train[:80]', 'train[80%:90%]','train[90%:]'], with_info=True, as_supervised=True)
# Display data
num_examples = metadata.splits['train'].num_examples
num_classes = metadata.features['label'].num_classes
print("exmaples: {}, classes: {}".format(num_examples, num_classes))

# Split data
BATCH_SIZE = 32
train_batches = train_img.shuffle(num_examples//4).map(format_img).batch(BATCH_SIZE).prefetch(1)
validate_batches = validate_img.map(format_img).batch(BATCH_SIZE).prefetch(1)
test_batches = test_img.map(format_img).batch(1)

# Display data shape
for img_batch, label_batch in train_batches.take(1):
    pass
img_batch.shape