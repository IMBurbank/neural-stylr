"""
Load Vgg19 Model Util

Functions to load imagenet-vgg-verydeep-19.mat model.
"""
import configparser
import scipy.io
import numpy as np
import tensorflow as tf


conf = configparser.ConfigParser()
conf.read('config.ini')
CONFIG = conf['vgg-19']


def load_vgg_model(path):
    """Load vgg-19 model graph.

    A model for the purpose of 'painting' the picture.
    Takes only the convolution layer weights and wrap using the TensorFlow
    Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
    the paper indicates that using AveragePooling yields better results.
    The last few fully connected layers are not used.

    Args:
    path: str path to pretrained model

    Returns:
    graph: VGG19 model graph with layer configuration
        Layers
            0:  conv1_1 (3, 3, 3, 64)
            1:  relu
            2:  conv1_2 (3, 3, 64, 64)
            3:  relu    
            4:  maxpool
            5:  conv2_1 (3, 3, 64, 128)
            6:  relu
            7:  conv2_2 (3, 3, 128, 128)
            8:  relu
            9:  maxpool
            10: conv3_1 (3, 3, 128, 256)
            11: relu
            12: conv3_2 (3, 3, 256, 256)
            13: relu
            14: conv3_3 (3, 3, 256, 256)
            15: relu
            16: conv3_4 (3, 3, 256, 256)
            17: relu
            18: maxpool
            19: conv4_1 (3, 3, 256, 512)
            20: relu
            21: conv4_2 (3, 3, 512, 512)
            22: relu
            23: conv4_3 (3, 3, 512, 512)
            24: relu
            25: conv4_4 (3, 3, 512, 512)
            26: relu
            27: maxpool
            28: conv5_1 (3, 3, 512, 512)
            29: relu
            30: conv5_2 (3, 3, 512, 512)
            31: relu
            32: conv5_3 (3, 3, 512, 512)
            33: relu
            34: conv5_4 (3, 3, 512, 512)
            35: relu
            36: maxpool
            37: fullyconnected (7, 7, 512, 4096)
            38: relu
            39: fullyconnected (1, 1, 4096, 4096)
            40: relu
            41: fullyconnected (1, 1, 4096, 1000)
            42: softmax
    """
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']
    

    def _weights(layer, expected_layer_name):
        """Return the weights and bias from the VGG model for a given layer."""
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    def _relu(conv2d_layer):
        """Return the RELU function wrapped over a TensorFlow layer.
        
        Expects a Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        """Return the Conv2D layer.
        
        Calculate using the weights, biases from the VGG model at 'layer'.
        """
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        """Return the Conv2D + RELU layer.
        
        Calculate using the weights, biases from the VGG model at 'layer'.
        """
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        """Return the AveragePooling layer."""
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Construct the graph model.
    graph = {}
    graph['input'] = tf.Variable(
        np.zeros((
            1,
            int(CONFIG['IMAGE_HEIGHT']),
            int(CONFIG['IMAGE_WIDTH']),
            int(CONFIG['COLOR_CHANNELS'])
        )),
        dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph