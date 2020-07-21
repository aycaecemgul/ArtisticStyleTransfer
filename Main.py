import cv2
import tensorflow as tf
import keras
from keras_preprocessing.image import load_img
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# displaying the content and style images

content_path = "content.jpeg"
style_path = "style.jpeg"

content = load_img(content_path)
style = load_img(style_path)


def display_images(content, style):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    ax1.imshow(content)
    ax1.set_title('Content')
    ax2.imshow(style)
    ax2.set_title('Style')
    plt.show()


# load and reshape the two images in arrays of numbers

def load_image(image_path):
    dim = 400
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [dim, dim])
    img = img[tf.newaxis, :]
    return img


# reshape and display images
content = load_image(content_path)
style = load_image(style_path)

# loading a pre-trained VGG19 model for extracting the features.

x = tf.keras.applications.vgg19.preprocess_input(content * 255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)

# Content layer
# or block4_conv2
content_layers = ['block4_conv2']

# Style layer
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# build the custom VGG model which will be composed of the specified layers.

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style * 255)

# Look at the statistics of each layer's output
# for name, output in zip(style_layers, style_outputs):
#     print(name)
#     print("  shape: ", output.numpy().shape)
#     print("  min: ", output.numpy().min())
#     print("  max: ", output.numpy().max())
#     print("  mean: ", output.numpy().mean())
#     print()


# #Defining a gram matrix

def gram_matrix(tensor):
    temp = tensor
    temp = tf.squeeze(temp)
    fun = tf.reshape(temp, [temp.shape[2], temp.shape[0] * temp.shape[1]])
    result = tf.matmul(temp, temp, transpose_b=True)
    gram = tf.expand_dims(result, axis=0)
    return gram


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


# Note that the content and style images are loaded in
# content and style variables respectively

extractor = StyleContentModel(style_layers, content_layers)
results = extractor(tf.constant(content))

# gradient descent
style_targets = extractor(style)['style']
content_targets = extractor(content)['content']

# Define a tf.Variable to contain the image to optimize.
# To make this quick, initialize it with the content image
# (the tf.Variable must be the same shape as the content image)
image = tf.Variable(content)

#Since this is a float image, define a function
# to keep the pixel values between 0 and 1:
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


opt = tf.optimizers.Adam(learning_rate=0.02)

# Custom weights for style and content updates
style_weight = 100  # 1e-2
content_weight = 10  # 1e4

# Custom weights for different style layers
style_weights = {'block1_conv1': 1.,
                 'block2_conv1': 0.8,
                 'block3_conv1': 0.5,
                 'block4_conv1': 0.3,
                 'block5_conv1': 0.1}


# The loss function to optimize
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


total_variation_weight=30

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

image = tf.Variable(content)

epochs = 5
steps_per_epoch = 200

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
  plt.imshow(np.squeeze(image.read_value(), 0))
  plt.title("Train step: {}".format(step))
  plt.show()

file_name = 'stylized-image.png'
tensor_to_image(image).save(file_name)

# try:
#   from google.colab import files
# except ImportError:
#    pass
# else:
#   files.download(file_name)