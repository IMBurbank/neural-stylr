import importlib
import sys
import tensorflow as tf
from utils.base import parse_args


def main(argv):
    model = importlib.import_module(FLAGS.model)
    
    model.main(FLAGS, argv)


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)