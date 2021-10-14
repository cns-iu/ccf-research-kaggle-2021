import random
from pathlib import Path

import cv2
import numpy as np
from utils import rgb2gray
import albumentations as albu
from PIL import Image


class AddGammaCorrection(albu.core.transforms_interface.ImageOnlyTransform):
    def __init__(self, p=.5, max_gray_val=255, gamma1=2.5, gamma2=4.0):
        """
        p: probability to apply whitening transform
        """
        super(AddGammaCorrection, self).__init__(p=p)
        self.max_gray_val = max_gray_val
        self.gamma1, self.gamma2 = gamma1, gamma2

    def _adgust_gamma(self, img, gamma):
        """Applies gamma correction.
            IMG, (h, w, 3)
        """
        assert img.shape[2] == 3  # c
        assert img.shape[0] == img.shape[1]  # hw

        inv_gamma = 1.0 / gamma
        table = np.array([((i / self.max_gray_val) ** inv_gamma) * self.max_gray_val
                          for i in np.arange(0, self.max_gray_val + 1)]).astype(np.uint8)
        return cv2.LUT(img, table).astype(np.uint8)

    def _combine_corrections(self, img):
        img1 = self._adgust_gamma(img, self.gamma1)
        img2 = self._adgust_gamma(img, self.gamma2)

        thresh = img.mean() - 0.8 * img.std()
        mask = img.mean(2) < thresh
        img2[mask] = img1[mask]
        return img2

    def apply(self, img, **params):
        return self._combine_corrections(img)


class _AddOverlayBase(albu.core.transforms_interface.ImageOnlyTransform):
    def __init__(self, get_overlay_fn, alpha=1, p=.5):
        """
        p: probability to apply blending transform
        d4_prob: probability to apply d4 transform
        """
        super(_AddOverlayBase, self).__init__(p=p)
        self.alpha = alpha
        if np.allclose(self.alpha, 0.0): raise Exception

        self.beta = 1 - alpha
        self.gamma = 0.0
        assert 0 <= self.alpha <= 1, f"Invalid alpha value equal to {self.alpha} (from 0.0 to 1.0)"
        self.get_overlay_fn = get_overlay_fn

    def d4(self, img, p=.5):
        if random.random() < p:
            tr = albu.Compose([albu.Flip(), albu.RandomRotate90()])
            return tr(image=img)['image']

        return img

    def _blend(self, image1, image2): return cv2.addWeighted(image1, self.alpha, image2, self.beta, self.gamma)

    def alpha_blend(self, image, aug_image):
        """
        IMAGE, (h, w, 3)
        AUG_IMAGE,  (h, w, 4) containing mask in last channel/band
        """
        assert image.shape[2] == 3 and aug_image.shape[2] == 4, (image.shape, aug_image.shape)  # c
        assert image.shape[:2] == aug_image.shape[:2]  , (image.shape, aug_image.shape)  # hw

        aug_image = self.d4(aug_image, p=1)
        rgb, mask = aug_image[...,:3], aug_image[...,3] > 0
        blended = self._blend(rgb, image)
        image[mask] = blended[mask]
        return image

    def apply(self, img, **params):
        return self.alpha_blend(img, self.get_overlay_fn())


class AddLightning(_AddOverlayBase):
    def __init__(self, imgs_path, crop_w, alpha=1, p=.5):
        super(AddLightning, self).__init__(get_overlay_fn=self.get_lightning, alpha=alpha, p=p)
        self.imgs = list(Path(imgs_path).rglob('*.png'))
        assert len(self.imgs) > 0, imgs_path
        self.crop_w = crop_w

    def _expander(self, img): 
        img = np.array(img)
        img = albu.RandomCrop(self.crop_w, self.crop_w)(image=img)['image']
        return img

    def get_lightning(self):
        img = Image.open(str(random.choice(self.imgs)))
        return self._expander(img)


class AddFakeGlom(_AddOverlayBase):
    def __init__(self, masks_path, crop_w, alpha=.7, p=.5, base_r=145, base_g=85, base_b=155, shift=30):
        """Default base_r, base_r, base_b and shift are chosen from hist."""
        super(AddFakeGlom, self).__init__(get_overlay_fn=self.get_glom, alpha=alpha, p=p)
        self.masks = list(Path(masks_path).rglob('*.png'))
        self.base_r, self.base_g, self.base_b = base_r, base_g, base_b
        self.shift = shift
        self.crop_w = crop_w

    def _expander(self, img):
        img = np.array(img)
        img = albu.RandomCrop(self.crop_w, self.crop_w)(image=img)['image']
        #print(img.shape)
        return img

    def _aug_with_rand_rgb(self, mask):
        """Returns aug_image shape of (h, w, 4) containing rgb and mask:
            - rgb pixel values are integers randomly drawn colour
            - mask pixel values are either 0 or 255
        """
        h, w, c = mask.shape  # hw1
        assert c == 3, f"Invalid number of channels, {c}"
        mask = np.expand_dims(mask[...,0], -1)

        shift_r, shift_g, shift_b = np.random.randint(-self.shift, self.shift, size=3)
        r = np.full((h, w, 1), self.base_r + shift_r, dtype=np.uint8)  # hw1
        g = np.full((h, w, 1), self.base_g + shift_g, dtype=np.uint8)  # hw1
        b = np.full((h, w, 1), self.base_b + shift_b, dtype=np.uint8)  # hw1
        return np.concatenate((r, g, b, mask), axis=2)  # hw4

    def get_glom(self):
        mask = Image.open(str(random.choice(self.masks)))  # hw1
        mask = self._expander(mask)
        return self._aug_with_rand_rgb(mask)  # hw4
