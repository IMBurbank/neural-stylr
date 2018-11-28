import argparse
import sys


def parse_args():
    """Parse input arguments.

    Args:
    None

    Returns:
    FLAGS: Parsed arguments
    unparsed: Unparsed arguments
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
        '-s',
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
        '--model',
        type=str,
        default='nst_vgg19',
        help='File path to the model to import. Default: nst_vgg19')
    parser.add_argument(
        '-p',
        '--pretrained_model',
        type=str,
        default='pretrained/imagenet-vgg-verydeep-19.mat',
        help='File path to the input pretained model.')
    parser.add_argument(
        '-c',
        '--content_image',
        type=str,
        default='images/input/default-content-image.png',
        help='File path to the input content image.')
    parser.add_argument(
        '-g',
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

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed