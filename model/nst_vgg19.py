import argparse
import glob
import logging
import os
import shutil
import sys
import tempfile
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from utils_img import *
from utils_vgg19 import *


STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


def parse_args(unparsed_args=None):
    """Parse input arguments.

    Args:
    unparsed_args: Unparsed arguments to add to args

    Returns:
    args: Parsed arguments
    """
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help(sys.stderr)
            sys.exit(1)

    parser = MyParser(
        description='Neural-style transfer using pretrained model vgg-19.')

    parser.add_argument(
        '-t',
        '--train_steps',
        type=int,
        default=200,
        help='Number of training steps. Default: 200')

    parser.add_argument(
        '-i',
        '--log_interval',
        type=int,
        default=20,
        help='Log metrics and save output image on interval. Default: 20')

    parser.add_argument(
        '-r',
        '--learning_rate',
        type=float,
        default=2.0,
        help='Learning rate. Default: 2.0')

    parser.add_argument(
        '-a',
        '--alpha',
        type=int,
        default=10,
        help='Content image weight factor. Default: 10')

    parser.add_argument(
        '-b',
        '--beta',
        type=int,
        default=40,
        help='Style image weight factor. Default: 40')

    parser.add_argument(
        '--noise_ratio',
        type=float,
        default=0.6,
        help='Random noise ratio in generated image. Default: 0.6')

    parser.add_argument(
        '-w',
        '--style_weights',
        nargs=5,
        type=float,
        default=[0.2, 0.2, 0.2, 0.2, 0.2],
        help='Layer style weights. Default: 0.2 0.2 0.2 0.2 0.2')

    parser.add_argument(
        '-l',
        '--output_layer',
        type=str,
        default='conv4_2',
        help='Layer to use for genreated image output. Default: conv4_2')

    parser.add_argument(
        '-m',
        '--input_model',
        type=str,
        default='pretrained/imagenet-vgg-verydeep-19.mat',
        help='File path to the input model.')

    parser.add_argument(
        '-c',
        '--content_image',
        type=str,
        default='images/input/default-content-image.png',
        help='File path to the input content image.')

    parser.add_argument(
        '-s',
        '--style_image',
        type=str,
        default='images/input/default-style-image.png',
        help='File path to the input style  image.')

    parser.add_argument(
        '-n',
        '--img_base_name',
        type=str,
        default='img',
        help='Output image base name. Default: img')

    parser.add_argument(
        '-o',
        '--output_img_dir',
        type=str,
        default='images/output/',
        help='File path to the input style image. Default: images/output/')

    parser.add_argument(
        '-d',
        '--drop_intermediate_images',
        action='store_true',
        help='Only save final generated image. Boolean toggle.')

    args = parser.parse_args(unparsed_args)

    return args


def compute_content_cost(a_C, a_G):
    """Compute content cost
    
    Args:
    a_C: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations
        representing content of the image C 
    a_G: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations
        representing content of the image G
    
    Returns: 
    J_content: tensor representing a scalar value, content cost
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(tf.transpose(a_C, perm=[3,0,1,2]), [n_C, -1])
    a_G_unrolled = tf.reshape(tf.transpose(a_G, perm=[3,0,1,2]), [n_C, -1])

    J_content = tf.divide(
        tf.reduce_sum((a_C_unrolled - a_G_unrolled) ** 2),
        4 * n_H * n_W * n_C)

    return J_content


def gram_matrix(A):
    """Compute gram matrix.

    Args:
    A: matrix of shape (n_C, n_H*n_W)
    
    Returns:
    Gram: Tensorflow matrix of shape (n_C, n_C), Gram matrix of A
    """
    Gram = tf.matmul(A, A, transpose_b=True)
    return Gram


def compute_layer_style_cost(a_S, a_G):
    """Compute cost of a style layer.

    Args:
    a_S: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations
        representing style of the image S 
    a_G: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations
        representing style of the image G
    
    Returns: 
    J_style_layer: tensor representing a scalar value, style cost for layer
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.reshape(tf.transpose(a_S, perm=[3,0,1,2]), [n_C, -1])
    a_G = tf.reshape(tf.transpose(a_G, perm=[3,0,1,2]), [n_C, -1])

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.cast(
        tf.divide(
            tf.reduce_sum((GS - GG) ** 2),
            4 * n_C ** 2 * (n_H * n_W) ** 2),
        tf.float32)
    
    return J_style_layer


def compute_style_cost(sess, model, style_profile):
    """Compute total style cost for chosen layers.
    
    Args:
    sess: Tensorflow session
    model: Tensorflow model
    style_profile: A python list containing:
        - the names of the layers we would like to extract style from
        - a coefficient for each of them
    
    Returns: 
    J_style: tensor representing a scalar value, total style cost
    """
    J_style = 0

    for layer_name, coeff in style_profile:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """Compute total cost.
    
    Args:
    J_content: tensor representing content cost
    J_style: tensor representing style cost
    alpha: int hyperparameter weighting the importance of the content cost
    beta: int hyperparameter weighting the importance of the style cost
    
    Returns:
    J: tensor representing a scalar value, total cost
    """
    J = alpha * J_content + beta * J_style
    return J


def transfer_style(sess,
                   model,
                   train_step,
                   J_content,
                   J_style,
                   J_total,
                   input_image,
                   num_iterations=200,
                   log_interval=20,
                   output_img_dir='output/',
                   output_img_base='img-xxxx'):
    """Run vgg-19 model to generate a NST image.

    Args:
    sess: Tensorflow session
    model: Tensorflow model
    J_content: tensor representing content cost
    J_style: tensor representing style cost
    J_total: tensor representing total cost
    input_image: Numpy array of an image
    num_iterations: int number of style-transfer steps
    log_interval: Log interval for metrics and saving intermediate images
    output_img_dir: str path to output image directory
    output_img_base: str base image name

    Returns:
    generated_image: Numpy array of generated image
    """
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i % log_interval == 0:
            Jt, Jc, Js = sess.run([J_total, J_content, J_style])
            logging.info("Iteration " + str(i) + " :")
            logging.info("total cost = " + str(Jt))
            logging.info("content cost = " + str(Jc))
            logging.info("style cost = " + str(Js))
            
            save_image(
                os.path.join(
                    output_img_dir,
                    '{}-step{}.png'.format(output_img_base, i)),
                generated_image)
    
    save_image(
        os.path.join(output_img_dir, '{}-out.png'.format(output_img_base)),
        generated_image)
    
    return generated_image


def main(unparsed_args=None):
    args = parse_args(unparsed_args)

    logging.basicConfig(
        level=logging.INFO,
        format=('%(levelname)s|%(asctime)s'
                '|%(pathname)s|%(lineno)d| %(message)s'),
        datefmt='%Y-%m-%dT%H:%M:%S',)
    logging.getLogger().setLevel(logging.INFO)
    logging.info(args)

    timestamp = int(time.time())
    output_img_base = '{}-{}'.format(args.img_base_name, timestamp)
    style_profile = list(zip(STYLE_LAYERS, args.style_weights))

    if not os.path.exists(args.output_img_dir):
        os.makedirs(args.output_img_dir)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    content_image = Image.open(args.content_image)
    content_image = resize_and_crop_image(content_image)
    content_image = reshape_and_normalize_image(content_image)
    style_image = Image.open(args.style_image)
    style_image = resize_and_crop_image(style_image)
    style_image = reshape_and_normalize_image(style_image)
    generated_image = generate_noise_image(content_image, args.noise_ratio)
    model = load_vgg_model(args.input_model)
    
    sess.run(model['input'].assign(content_image))

    out = model[args.output_layer]
    a_C = sess.run(out)
    a_G = out

    sess.run(model['input'].assign(style_image))

    J_content = compute_content_cost(a_C, a_G)
    J_style = compute_style_cost(sess, model, style_profile)
    J = total_cost(J_content, J_style, args.alpha, args.beta)
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_step = optimizer.minimize(J)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            transfer_style(
                sess=sess,
                model=model,
                train_step=train_step,
                J_content=J_content,
                J_style=J_style,
                J_total=J,
                input_image=generated_image,
                num_iterations=args.train_steps,
                log_interval=args.log_interval,
                output_img_dir=tmpdir,
                output_img_base=output_img_base)
        except OSError as err:
            logging.error(str(err))
        finally:
            if args.drop_intermediate_images:
                output_images = glob.glob(tmpdir + '*.png')
                latest_image = max(output_images, key=os.path.getctime)
                shutil.copy2(
                    os.path.join(tmpdir, latest_image),
                    os.path.join(args.output_img_dir, latest_image))
            else:
                shutil.copytree(
                    tmpdir,
                    os.path.join(args.output_img_dir, output_img_base))

    logging.info('Neural-style transfer complete.')


if __name__ == '__main__':
    main()