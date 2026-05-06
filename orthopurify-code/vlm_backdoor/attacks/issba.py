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
    def __init__(self, model_path, secret, size, residual_alpha=None):
        BCH_POLYNOMIAL = 137
        BCH_BITS = 5
        self.size = size
        # residual 放大系数：α=1.0 等同于原版 ISSBA（返回 hidden_img）
        # α>1.0 放大隐写残差以强化触发信号（适配 VLM 高分辨率输入）
        # 支持通过环境变量 ISSBA_RESIDUAL_ALPHA 覆盖
        if residual_alpha is None:
            residual_alpha = float(os.environ.get("ISSBA_RESIDUAL_ALPHA", "1.0"))
        self.residual_alpha = float(residual_alpha)
        print(f"[ISSBA] residual_alpha = {self.residual_alpha}")

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
        image_np = np.array(image, dtype=np.float32) / 255.0

        feed_dict = {
            self.input_secret: [self.secret],  # [B, secret_len]
            self.input_image:  [image_np],     # [B, H, W, C]
        }
        hidden_img, residual = self.sess.run(
            [self.output_stegastamp, self.output_residual],
            feed_dict=feed_dict
        )
        # α=1.0 时等价于返回 hidden_img
        # α>1.0 时放大残差：从 hidden_img 反推实际扰动量，保证语义正确
        if abs(self.residual_alpha - 1.0) < 1e-6:
            out = hidden_img[0]
        else:
            actual_residual = hidden_img[0] - image_np  # 真实的实际扰动
            out = image_np + self.residual_alpha * actual_residual
        out = np.clip(out, 0.0, 1.0)
        out_uint8 = (out * 255.0).astype(np.uint8)
        return Image.fromarray(out_uint8)
