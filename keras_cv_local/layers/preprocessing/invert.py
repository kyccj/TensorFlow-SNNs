import tensorflow as tf

try:
    from keras_cv.layers.preprocessing.base_image_augmentation_layer import BaseImageAugmentationLayer
    from keras_cv.utils import preprocessing
except (ImportError, ModuleNotFoundError):
    from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import BaseImageAugmentationLayer
    try:
        from keras_cv.src.utils import preprocessing
    except ImportError:
        preprocessing = None


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class Invert(BaseImageAugmentationLayer):
    def __init__(
            self,
            factor,
            seed=None,
            **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.seed = seed

    def augment_image(self,image,transformation=None,**kwargs):
        image = tf.convert_to_tensor(image)
        return 255-image
