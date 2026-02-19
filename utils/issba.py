import os

if os.environ.get("ISSBA_FORCE_CPU", "1") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import bchlib
import tensorflow as tf
from PIL import Image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

tf.compat.v1.disable_eager_execution()

class issbaEncoder(object):
    def __init__(self, model_path, secret, size):
        BCH_POLYNOMIAL = 137
        BCH_BITS = 5
        self.size = size

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.device_count["GPU"] = 0

        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

        with self.graph.as_default():
            model = tf.compat.v1.saved_model.loader.load(
                self.sess, [tag_constants.SERVING], model_path
            )

            input_secret_name = model.signature_def[
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            ].inputs["secret"].name
            input_image_name = model.signature_def[
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            ].inputs["image"].name

            self.input_secret = self.graph.get_tensor_by_name(input_secret_name)
            self.input_image  = self.graph.get_tensor_by_name(input_image_name)

            output_stegastamp_name = model.signature_def[
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            ].outputs["stegastamp"].name
            output_residual_name = model.signature_def[
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            ].outputs["residual"].name

            self.output_stegastamp = self.graph.get_tensor_by_name(output_stegastamp_name)
            self.output_residual   = self.graph.get_tensor_by_name(output_residual_name)

        bch = bchlib.BCH(t=BCH_BITS, prim_poly=BCH_POLYNOMIAL)
        if len(secret) > 7:
            raise ValueError("Can only encode 56 bits (7 ASCII chars) with ECC")

        data = bytearray(secret + " " * (7 - len(secret)), "utf-8")
        ecc = bch.encode(data)
        packet = data + ecc
        packet_binary = "".join(format(x, "08b") for x in packet)
        secret_bits = [int(x) for x in packet_binary]
        secret_bits.extend([0, 0, 0, 0])  # pad
        self.secret = secret_bits

        try:
            print("[ISSBA] TF physical GPUs:", tf.config.list_physical_devices("GPU"))
        except Exception:
            pass

    def __call__(self, image):
        image = image.resize((224, 224))
        image = np.array(image, dtype=np.float32) / 255.0

        feed_dict = {
            self.input_secret: [self.secret],  # [B, secret_len]
            self.input_image:  [image],        # [B, H, W, C]
        }
        hidden_img, residual = self.sess.run(
            [self.output_stegastamp, self.output_residual],
            feed_dict=feed_dict
        )
        hidden_img = (hidden_img[0] * 255.0).astype(np.uint8)
        return Image.fromarray(hidden_img)
