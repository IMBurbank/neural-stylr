import pytest
import unittest
import numpy as np
import tensorflow as tf
from PIL import Image
from model.nst_vgg19 import (
    compute_content_cost,
    gram_matrix,
    compute_layer_style_cost,
    total_cost,
)


class TestComputeContentCost(unittest.TestCase):
    def test_content_cost_value(self):
        content_cost = 6.7655926
        tf.reset_default_graph()
        with tf.Session() as test:
            tf.set_random_seed(1)
            a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
            a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
            J_content = compute_content_cost(a_C, a_G).eval()
        self.assertAlmostEqual(J_content, content_cost)


class TestGramMatrix(unittest.TestCase):
    def test_gram_matrix_values(self):
        expected = np.array([
                [ 6.422305, -4.429122, -2.096682],
                [-4.429122, 19.465837, 19.563871],
                [-2.096682, 19.563871, 20.686462]
            ],
            dtype=np.float32)
        tf.reset_default_graph()
        with tf.Session() as test:
            tf.set_random_seed(1)
            A = tf.random_normal([3, 2*1], mean=1, stddev=4)
            GA = gram_matrix(A).eval()
        self.assertTrue(np.array_equal(GA, expected))


class TestComputeLayerStyleCost(unittest.TestCase):
    def test_layer_style_cost_value(self):
        expected = 9.190277100
        tf.reset_default_graph()
        with tf.Session() as test:
            tf.set_random_seed(1)
            a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
            a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
            J_style_layer = compute_layer_style_cost(a_S, a_G).eval()
        self.assertAlmostEqual(J_style_layer, expected)


class TestTotalCost(unittest.TestCase):
    def test_total_cost_value(self):
        expected = 35.34667875478276
        tf.reset_default_graph()
        with tf.Session() as test:
            np.random.seed(3)
            J_content = np.random.randn()
            J_style = np.random.randn()
            J = total_cost(J_content, J_style)
        self.assertAlmostEqual(J, expected)
