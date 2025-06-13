# python3.8
"""Implements image decoding."""

import cv2
import numpy as np

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    import nvidia.dali.math as dmath
except ImportError:
    fn = None

from .utils.formatting_utils import format_image
from .base_transformation import BaseTransformation

__all__ = ['Decode']


class Decode(BaseTransformation):
    """Decodes image buffers to images.

    Args:
        image_channels: Number of channels of the decoded image. This field is
            only used for the function `self._DALI_forward()`. The function
            `self._CPU_forward()` will determine the image channels
            automatically. (default: 3)
        return_square: Whether to return a square image by keeping the length
            of the image short side and cropping along the long side. If set
            as `True`, the decoding and cropping operations will be executed on
            CPU for DALI forwarding. (default: False)
        center_crop: This field only takes effect when `return_square` is set
            as `True`. It determines whether to centrally crop the image along
            the long side. (default: True)

    Raises:
        ValueError: If the `image_channels` is not supported, i.e., not one of
            `1` (GRAY), `3` (RGB), `4` (RGBA).
    """

    def __init__(self, image_channels=3, return_square=False, center_crop=True):
        super().__init__(support_dali=(fn is not None))

        if image_channels == 1:
            self.image_type = 'GRAY'
        elif image_channels == 3:
            self.image_type = 'RGB'
        elif image_channels == 4:
            self.image_type = 'RGBA'
        else:
            raise ValueError(f'Invalid image channels: `{image_channels}`!\n'
                             f'Channels allowed: 1 (GRAY), 3 (RGB), 4 (RGBA).')

        self.image_channels = image_channels
        self.return_square = return_square
        self.center_crop = center_crop

        if self.image_type == 'RGBA':  # DALI dose not support RGBA format.
            self._support_dali = False

    
    def __call__(self, data, crop_pos_ = 0.5, use_dali=False):
        """Transforms the input data with the proper manner.

        Basically, this function chooses between `self._CPU_forward()` and
        `self._DALI_forward()`. In addition, this function handles the case
        where `data` is not a list, such that `self._CPU_forward()` and
        `self._DALI_forward()` only need to consider list input.

        Args:
            use_dali: Whether to use `self._DALI_forward()` for forwarding or
                not. (default: False)
        """
        is_input_list = True

        if not isinstance(data, (list, tuple)):
            is_input_list = False
            data = [data]

        if use_dali and self.support_dali and dali_pipeline is not None:
            outputs = self._DALI_forward(data)
        else:
            outputs = self._CPU_forward(data, crop_pos_)

        if not is_input_list and isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs


    def _CPU_forward(self, data, crop_pos_ = -1):
        if self.center_crop:
            crop_pos = 0.5
        elif crop_pos_ == -1:
            crop_pos = np.random.uniform()
        else:
            crop_pos = crop_pos_

        outputs = []
        for buffer in data:
            image = format_image(cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED))
            height, width = image.shape[:2]
            if not self.return_square or height == width:
                outputs.append(image)
                continue
            crop_size = min(height, width)
            y = int((height - crop_size) * crop_pos)
            x = int((width - crop_size) * crop_pos)
            outputs.append(np.ascontiguousarray(
                image[y:y + crop_size, x:x + crop_size]))
        return outputs

    def _DALI_forward(self, data):
        if self.image_type == 'GRAY':
            output_type = types.GRAY
        elif self.image_type == 'RGB':
            output_type = types.RGB
        else:
            raise NotImplementedError(f'Not implemented image type '
                                      f'`{self.image_type}`!')

        # Use `mixed` device for decoding if no cropping is needed.
        if not self.return_square:
            outputs = []
            for buffer in data:
                image = fn.decoders.image(buffer,
                                          device='mixed',
                                          output_type=output_type)
                outputs.append(image.gpu())
            return outputs

        # Decode image with CPU if cropping with dynamic size.
        if self.center_crop:
            crop_pos = 0.5
        else:
            crop_pos = fn.random.uniform(range=(0, 1))
        outputs = []
        for buffer in data:
            image = fn.decoders.image(buffer,
                                      device='cpu',
                                      output_type=output_type)
            image_shape = fn.shapes(image)
            height = fn.slice(image_shape, 0, 1, axes=[0], dtype=types.FLOAT)
            width = fn.slice(image_shape, 1, 1, axes=[0], dtype=types.FLOAT)
            crop_length = dmath.min(height, width)
            image = fn.crop(image,
                            crop_pos_x=crop_pos,
                            crop_pos_y=crop_pos,
                            crop_w=crop_length,
                            crop_h=crop_length,
                            out_of_bounds_policy='error')
            outputs.append(image.gpu())
        return outputs
