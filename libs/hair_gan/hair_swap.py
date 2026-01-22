import typing as tp
from collections import defaultdict
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.io import read_image, ImageReadMode

from hair_gan.models.Alignment import Alignment
from hair_gan.models.Blending import Blending
from hair_gan.models.Embedding import Embedding
from hair_gan.models.Net import Net
from hair_gan.utils.image_utils import equal_replacer
from hair_gan.utils.seed import seed_setter
from hair_gan.utils.shape_predictor import align_face
from hair_gan.utils.time import bench_session

TImage = tp.TypeVar('TImage', torch.Tensor, Image.Image, np.ndarray)
TPath = tp.TypeVar('TPath', Path, str)
TReturn = tp.TypeVar('TReturn', torch.Tensor, dict[str, torch.Tensor])


class HairFast:
    """
    HairFast implementation with hairstyle transfer interface
    """

    def __init__(self, **kwargs):
        from argparse import Namespace
        self.args = Namespace(**kwargs)

        self.net = Net(self.args)
        self.embed = Embedding(self.args, net=self.net)
        self.align = Alignment(
            self.args, self.embed.get_e4e_embed, net=self.net)
        self.blend = Blending(self.args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        # Embedding stage
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        # Alignment stage
        align_shape = self.align.align_images(
            'face', 'shape', name_to_embed, **kwargs)

        # Shape Module stage for blending
        if shape is not color:
            align_color = self.align.shape_module(
                'face', 'color', name_to_embed, **kwargs)
        else:
            align_color = align_shape

        # Blending and Post Process stage
        final_image = self.blend.blend_images(
            align_shape, align_color, name_to_embed, **kwargs)

        return final_image

    def _process_image(self, path_to_images):
        images = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in path_to_images.values():
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(
                        str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        return images

    def swap(
        self,
            face_img: TImage | TPath,
            shape_img: TImage | TPath,
            color_img: TImage | TPath,
            predictor=None,
            benchmark=False,
            align=False,
            seed=None,
            exp_name=None,
            **kwargs
    ) -> TReturn:
        """
        Run HairFast on the input images to transfer hair shape and color to the desired images.
        :param face_img:  face image in Tensor, PIL Image, array or file path format
        :param shape_img: shape image in Tensor, PIL Image, array or file path format
        :param color_img: color image in Tensor, PIL Image, array or file path format
        :param benchmark: starts counting the speed of the session
        :param align:     for arbitrary photos crops images to faces
        :param seed:      fixes seed for reproducibility, default 3407
        :param exp_name:  used as a folder name when 'save_all' model is enabled
        :return:          returns the final image as a Tensor
        """
        images = self._process_images([face_img, shape_img, color_img])

        if align:
            images = align_face(images, predictor=predictor)

        # Reference the same image to reduce ram usage if they are similar
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(
            *images,
            seed=seed,
            benchmark=benchmark,
            exp_name=exp_name,
            **kwargs
        )

        if align:
            return {
                "final_image": final_image,
                "aligned_face": images[0],
                "aligned_hair": images[1],
                "aligned_color": images[2],
            }

        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)
