import glob
import os
import shutil
import sys
import tempfile
import time
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image
from utils.base import parse_args
from utils.model_vgg19 import load_vgg_model
from utils.image import (
    generate_noise_image,
    resize_and_crop_image,
    reshape_and_normalize_image,
    save_image
)


tf.logging.set_verbosity(tf.logging.INFO)
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
FLAGS = None



def compute_content_cost(a_C, a_G):
    """Compute content cost.
    
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
            tf.logging.info("Iteration " + str(i) + " :")
            tf.logging.info("total cost = " + str(Jt))
            tf.logging.info("content cost = " + str(Jc))
            tf.logging.info("style cost = " + str(Js))
            
            save_image(
                os.path.join(
                    output_img_dir,
                    '{}-step{}.png'.format(output_img_base, i)),
                generated_image)
    
    save_image(
        os.path.join(output_img_dir, '{}-out.png'.format(output_img_base)),
        generated_image)
    
    return generated_image


def main(args=None, argv=sys.argv):
    tf.logging.info(args)

    if args is None:
        if FLAGS is not None:
            args = FLAGS
        else:
            tf.logging.error('Error: Model requires args.')
            sys.exit(1)

    if not os.path.exists(args.output_img_dir):
        os.makedirs(args.output_img_dir)

    timestamp = int(time.time())
    output_img_base = '{}-{}'.format(args.img_base_name, timestamp)
    style_profile = list(zip(STYLE_LAYERS, args.style_weights))

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    content_image = Image.open(args.content_image)
    content_image = resize_and_crop_image(content_image)
    content_image = reshape_and_normalize_image(content_image)
    style_image = Image.open(args.style_image)
    style_image = resize_and_crop_image(style_image)
    style_image = reshape_and_normalize_image(style_image)
    generated_image = generate_noise_image(content_image, args.noise_ratio)
    model = load_vgg_model(args.pretrained_model)

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
            tf.logging.error(str(err))
        finally:
            img_dir = os.path.join(args.output_img_dir, output_img_base)
            if args.drop_intermediate_images:
                output_images = glob.glob(tmpdir + '*.png')
                latest_image = max(output_images, key=os.path.getctime)
                os.makedirs(img_dir)
                shutil.copy2(
                    os.path.join(tmpdir, latest_image),
                    os.path.join(args.output_img_dir, latest_image))
            else:
                shutil.copytree(tmpdir, img_dir)

    yaml_stream = open(os.path.join(img_dir, 'config.yaml'), 'w')
    yaml.dump(args, yaml_stream)
    tf.logging.info(
        'Neural-style transfer complete. Image(s) transferred to {}'.format(
            img_dir))


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)