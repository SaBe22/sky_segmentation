import os

import cv2
import numpy as np

# Default parameters for grab cut segmentation
DEFAULT_RESIZE_IMG_WIDTH: int = 512
DEFAULT_RESIZE_IMG_HEIGHT: int = 512
DEFAULT_BLUE_HUE_MIN_VALUE: int = 100
DEFAULT_BLUE_HUE_MAX_VALUE: int = 130
DEFAULT_KERNEL_SIZE_MASK_ERODE: int = 30
DEFAULT_MAX_DISTANCE: float = 30

class SegmentationGrabCut:
    """
    Perform sky segmentation using the GrabCut algorithm

    Attributes:
        resize_width (int): The width to resize the image to.
        resize_height (int): The height to resize the image to.
        blue_hue_min_value (int): The minimum hue value for blue color extraction
        blue_hue_max_value (int): The maximum hue value for blue color extraction.
        kernel_size_mask_erode (int): The kernel size to use for eroding the mask.
        max_distance (float): Maximum distance in pixels from the initial foreground estimate,
            pixels beyond this distance are considered as background.
    """

    def __init__(self, config: dict = None) -> None:
        """
        Initializes the SegmentationGrabCut class with the given configuration parameters.
        Args:
            config (dict, optional): A dictionary of configuration parameters.
                Defaults to an empty dictionary.
        """
        if config is None:
            config = {}
        self.resize_width = config.get("resize_width", DEFAULT_RESIZE_IMG_WIDTH)
        self.resize_height = config.get("resize_height", DEFAULT_RESIZE_IMG_HEIGHT)
        self.blue_hue_min_value = config.get("blue_hue_min_value", DEFAULT_BLUE_HUE_MIN_VALUE)
        self.blue_hue_max_value = config.get("blue_hue_max_value", DEFAULT_BLUE_HUE_MAX_VALUE)
        self.kernel_size_mask_erode = config.get("kernel_size_mask_erode",
            DEFAULT_KERNEL_SIZE_MASK_ERODE)
        self.max_distance = config.get("max_distance", DEFAULT_MAX_DISTANCE)

    def segment(self, image_path: str) -> np.ndarray:
        """
        Segment the sky region in the image using the GrabCut algorithm.

        Args:
            image_path (str): path to the input image

        Returns:
            np.ndarray: Binary mask of the sky region after GrabCut segmentation.
        """
        if not os.path.exists(image_path):
            raise ValueError(f"{image_path} is not a valid image_path {os.getcwd()}")

        image = cv2.imread(image_path)
        # Resize the image to a smaller size to speed up processing
        resized = cv2.resize(image, (self.resize_width, self.resize_height),
            interpolation=cv2.INTER_AREA)
        blue_color_mask = self.blue_color_extraction(resized)
        preprocessed_mask = self.preprocess_mask(blue_color_mask)

        # Initialize the foreground and background models for GrabCut
        background_model = np.zeros((1, 65), np.float64) # internal array of cv2.grabcut method
        foreground_model = np.zeros((1, 65), np.float64)

        # Apply GrabCut to refine the mask
        mask, background_model, foreground_model = cv2.grabCut(
            resized,
            preprocessed_mask,
            None,
            background_model,
            foreground_model,
            iterCount=5,
            mode=cv2.GC_INIT_WITH_MASK
        )

        # Get the foreground mask
        foreground_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        return foreground_mask

    def blue_color_extraction(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Initialize a sky mask based on the Hue value of the HSV color space.

        Args:
            image_path (np.ndarray): input image

        Returns:
            np.ndarray: A binary mask of the blue regions in the image.
        """

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract the H channel
        H = hsv[:, :, 0]

        # Threshold the H channel to create a mask of blue pixels
        blue_color_mask = cv2.inRange(H, self.blue_hue_min_value, self.blue_hue_max_value)

        # Apply a median filter to remove isolated pixels or small regions
        blue_color_mask = cv2.medianBlur(blue_color_mask, 5)

        return blue_color_mask

    def preprocess_mask(
            self,
            mask: np.ndarray,
        ) -> np.ndarray:
        """
        Improve the sky mask with morphological operators.

        Args:
            mask (np.ndarray): Binary mask of the sky region.
        Returns:
            np.ndarray: The processed binary mask compatible with grab cut segmentation.
        """
        # Note: the mask is inverted (using ~) because distanceTransform considers black pixels
        # to be part of the object and white pixels to be part of the background.
        dist_transform = cv2.distanceTransform(~mask, cv2.DIST_L2, 5)
        mask_preprocess = np.ones_like(mask) * cv2.GC_PR_BGD
        mask_preprocess[dist_transform > self.max_distance] = cv2.GC_BGD
        kernel = np.ones(
            (self.kernel_size_mask_erode, self.kernel_size_mask_erode),
            np.uint8
        )
        mask_erode = cv2.erode(mask, kernel, iterations=1)
        mask_preprocess[mask_erode == 255] = cv2.GC_FGD
        mask_diff = mask - mask_erode
        mask_preprocess[mask_diff == 255] = cv2.GC_PR_FGD

        return mask_preprocess
