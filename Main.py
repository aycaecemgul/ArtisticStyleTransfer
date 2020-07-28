from os import path
from os.path import isfile, join
import cv2
import os
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
    dim = 450
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [dim, dim])
    img = img[tf.newaxis, :]
    return img


# reshape and display images
content = load_image(content_path)
style = load_image(style_path)


# Content layer
# or block4_conv2
content_layers = ['block4_conv2']

"""input_1
block1_conv1
block1_conv2
block1_pool
block2_conv1
block2_conv2
block2_pool
block3_conv1
block3_conv2
block3_conv3
block3_conv4
block3_pool
block4_conv1
block4_conv2
block4_conv3
block4_conv4
block4_pool
block5_conv1
block5_conv2
block5_conv3
block5_conv4
block5_pool"""

# Style layer
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

#our model
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style * 255)

# # Look at the statistics of each layer's output
# # for name, output in zip(style_layers, style_outputs):
# #     print(name)
# #     print("  shape: ", output.numpy().shape)
# #     print("  min: ", output.numpy().min())
# #     print("  max: ", output.numpy().max())
# #     print("  mean: ", output.numpy().mean())
#     print()


# #Defining a gram matrix

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


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

image = tf.Variable(content)

#Since this is a float image, define a function
# to keep the pixel values between 0 and 1:
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Custom weights for style and content updates
style_weight = 200
content_weight = 1e4




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
#
#THIS WAS 30
total_variation_weight=800
#
@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight * tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

image = tf.Variable(content)

epochs = 5
steps_per_epoch = 400
step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
  plt.imshow(np.squeeze(image.read_value(), 0))
  plt.title("Train step: {}".format(step))
  plt.show()
  # file_name = 'stylized-image' + "/%#05d.jpg" % (step +1) +"frame.png"
  # tensor_to_image(image).save(file_name)

# def transfer_to_frames(frame):
#     for i in range(frame):
#         image = tf.Variable(frame)
#         train_step(image)
#         # file_name = 'stylized-frame' + "/%#05d.jpg" % (i).png'
#         tensor_to_image(image).save(file_name)

def video_to_frames(input_loc, output_loc):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        if (count % 4 == 1):
            cv2.imwrite(output_loc + "/%#05d.jpg" % (count + 1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length - 1)):
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            break

# if __name__ == "__main__":
#     input_loc = "video.mp4"
#     output_loc = "C:\\Users\\aycae\\PycharmProjects\\Internship\\Frames"
#     video_to_frames(input_loc, output_loc)


