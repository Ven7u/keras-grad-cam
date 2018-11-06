import json

from keras import Model
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
from tensorflow.python.keras.utils import get_file
import os
import pandas as pd


CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
OUT_FOLDER = "./out"


def get_image_base_name(image_url):
    n = os.path.basename(image_url).split(".")[0]
    return n.replace('_', '')






def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')

    #print("modify_backprop")
    #print(model.summary())
    #print(new_model.summary())
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def _compute_gradients(tensor, var_list):
  grads = tf.gradients(tensor, var_list)
  return [grad if grad is not None else tf.zeros_like(var)
          for var, grad in zip(var_list, grads)]

def grad_cam(input_model, image, category_index, layer_name):

    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    #print(input_model.output)
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)

    #print(input_model.summary())
    #print(model.summary())

    loss = K.sum(model.output)
    conv_output = [l for l in model.layers if l.name is layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def run_grad_cam(model, predicted_class, class_info, out_folder=OUT_FOLDER):
    cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
    cv2.imwrite(os.path.join(out_folder, "{}-{}-gradcam.jpg".format(image_id, class_info[1])), cam)

    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp')
    saliency_fn = compile_saliency_function(guided_model)
    saliency = saliency_fn([preprocessed_input, 0])
    gradcam = saliency[0] * heatmap[..., np.newaxis]
    cv2.imwrite(os.path.join(out_folder, "{}-{}-guided_gradcam.jpg".format(image_id, class_info[1])), deprocess_image(gradcam))



def get_classes_path():
    fpath = get_file(
        'imagenet_class_index.json',
        CLASS_INDEX_PATH,
        cache_subdir='models',
        file_hash='c2c37ea517e94d9795004a39431a14cb')

    return fpath

def get_classes():
    fpath = get_classes_path()

    with open(fpath) as f:
        return json.load(f)

def get_class_by_name(class_name):

    CLASS_INDEX = get_classes()
    for class_id, class_details in CLASS_INDEX.items():
        # name = str.replace(class_name, "_", " ")
        if class_name in class_details:
            return int(class_id), class_details[1], class_details[0]

    raise Exception("Impossible to find class {0} in Imagenet".format(class_name))


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

image_id = ""

if __name__ == '__main__':
    image_id = get_image_base_name(sys.argv[1])

    out_folder = os.path.join(OUT_FOLDER, image_id)
    create_folder(out_folder)

    preprocessed_input = load_image(sys.argv[1])

    model = VGG16(weights='imagenet')

    predictions = model.predict(preprocessed_input)
    top_10 = decode_predictions(predictions, 10)[0]

    top_10 = [(t[0], t[1], t[2], get_class_by_name(t[1])[0]) for t in top_10]

    top_10_df = pd.DataFrame(top_10)
    print(top_10_df)
    top_10_df.to_csv(os.path.join(out_folder, "top_10.csv"))

    for top_1 in top_10:
        print('Predicted class:')
        print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
        predicted_class = top_1[3]

        run_grad_cam(model, predicted_class, top_1, out_folder)

