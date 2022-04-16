import tensorflow_addons as tfa
import tensorflow as tf


class Loss:
    def __init__(self, cfg):
        self.semantic_loss = tf.keras.losses.MeanSquaredError()
        self.alpha_loss = tf.keras.losses.MeanAbsoluteError()
        self.fusion_loss = tf.keras.losses.MeanAbsoluteError()
        self.semantic_mult = cfg.project.training.semantic_mult
        self.alpha_mult = cfg.project.training.alpha_mult
        self.fusion_mult = cfg.project.training.fusion_mult

    def __call__(
        self, image, target, output_fusion, output_global, output_local, trimap
    ):
        target = tf.expand_dims(target, axis=-1)
        target_global = tf.image.resize(target, output_global.shape[1:3])
        target_global = tfa.image.gaussian_filter2d(target_global, sigma=0.8)
        semantic_loss = self.semantic_loss(target_global, output_global)
        pred_detail = tf.where(
            tf.math.logical_or(trimap < 0.1, trimap > 0.9),
            trimap,
            output_local[..., 0],
        )
        gt_detail = tf.where(
            tf.math.logical_or(trimap < 0.1, trimap > 0.9),
            trimap,
            target[..., 0],
        )
        alpha_loss = self.alpha_loss(pred_detail, gt_detail)
        pred_boundary_matte = tf.where(
            tf.math.logical_or(trimap < 0.1, trimap > 0.9),
            trimap,
            output_fusion[..., 0],
        )
        fusion_loss = self.fusion_loss(
            output_fusion, target
        ) + 4 * self.fusion_loss(pred_boundary_matte, target)
        fusion_comp_loss = self.fusion_loss(
            tf.multiply(image, output_fusion), tf.multiply(image, target)
        )
        pred_boundary_matte = tf.expand_dims(pred_boundary_matte, -1)
        fusion_comp_loss += 4 * self.fusion_loss(
            tf.multiply(image, pred_boundary_matte), tf.multiply(image, target)
        )
        fusion_loss = fusion_loss + fusion_comp_loss
        loss = self.semantic_mult * semantic_loss + self.alpha_mult * alpha_loss + self.fusion_mult *fusion_loss
        return loss, semantic_loss, alpha_loss, fusion_loss
