import os
import pytest
import tempfile
import unittest
import numpy as np
from PIL import Image
from utils.image import (
    generate_noise_image,
    resize_and_crop_image,
    reshape_and_normalize_image,
    save_image
)

class TestGenerateNoiseImage(unittest.TestCase):
    def test_is_callable(self):
        self.assertTrue(callable(generate_noise_image), msg='Is callable')

    def test_returns_same_shape_np_array(self):
        content_image = np.full((1, 300, 400, 3), fill_value=1)
        noise_ratio = 0.5
        input_image = generate_noise_image(
            content_image=content_image, noise_ratio=noise_ratio)
        self.assertIsInstance(input_image, np.ndarray, msg='Is np.array')
        self.assertSequenceEqual(
            content_image.shape, input_image.shape, msg='Output same shape')

    def test_returns_same_array_when_no_noise(self):
        content_image = np.full((1, 300, 400, 3), fill_value=1)
        noise_ratio = 0
        input_image = generate_noise_image(
            content_image=content_image, noise_ratio=noise_ratio)
        self.assertTrue(
            np.array_equal(content_image, input_image),
            msg='Same array when no noise')

    def test_returns_diff_array_when_noise(self):
        content_image = np.full((1, 300, 400, 3), fill_value=1)
        noise_ratio = 0.5
        input_image = generate_noise_image(
            content_image=content_image, noise_ratio=noise_ratio)
        self.assertFalse(
            np.array_equal(content_image, input_image),
            msg='Diff array with noise')


class TestResizeAndCropImage(unittest.TestCase):
    def test_image_unchanged_when_default_size(self):
        in_img = Image.new('RGB', (300,400))
        out_img = resize_and_crop_image(image=in_img, size=(300,400))
        self.assertTrue(
            np.array_equal(np.array(in_img), out_img),
            msg='Unchanged when default size')

    def test_shape_bigger_image_same_aspect(self):
        shape = (300, 400, 3)
        h, w, _ = shape
        in_img = Image.new('RGB', (600,800))
        out_img = resize_and_crop_image(image=in_img, size=(w,h))
        self.assertSequenceEqual(
            out_img.shape, shape,
            msg='Resized when bigger - same aspect ratio')

    def test_shape_smaller_image_same_aspect(self):
        shape = (300, 400, 3)
        h, w, _ = shape
        in_img = Image.new('RGB', (150,200))
        out_img = resize_and_crop_image(image=in_img, size=(w,h))
        self.assertSequenceEqual(
            out_img.shape, shape,
            msg='Resized when smaller - same aspect ratio')

    def test_shape_bigger_image_diff_aspect(self):
        shape = (300, 400, 3)
        h, w, _ = shape
        in_img = Image.new('RGB', (600,900))
        out_img = resize_and_crop_image(image=in_img, size=(w,h))
        self.assertSequenceEqual(
            out_img.shape, shape,
            msg='Resized when bigger - different aspect ratio')

    def test_shape_smaller_image_diff_aspect(self):
        shape = (300, 400, 3)
        h, w, _ = shape
        in_img = Image.new('RGB', (150,175))
        out_img = resize_and_crop_image(image=in_img, size=(w,h))
        self.assertSequenceEqual(
            out_img.shape, shape,
            msg='Resized when smaller - different aspect ratio')


class TestReshapeAndNormalizeImage(unittest.TestCase):
    def test_reshape(self):
        shape = (300, 400, 3)
        reshape = (1, 300, 400, 3)
        in_img = np.full(shape, fill_value=100)
        out_img = reshape_and_normalize_image(in_img)
        self.assertSequenceEqual(out_img.shape, reshape, msg='Reshape')

    def test_means_subtracted_in_rgb_range(self):
        shape = (300, 400, 3)
        in_img = np.full(shape, fill_value=255)
        out_img = reshape_and_normalize_image(in_img)
        self.assertTrue(
            (out_img >= 0).all() and (out_img <= 255).all(),
            msg='Subtracted means between 0 and 255')
