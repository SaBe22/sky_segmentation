import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import SegmentationModel
from segmentation_grabcut import SegmentationGrabCut
# Default model initialization parameters
from utils import (DEFAULT_IMG_HEIGHT, DEFAULT_IMG_WIDTH, DEFAULT_MODEL_NAME,
                   DEFAULT_PRETRAINED_WEIGHTS, NORMALIZE_MEAN, NORMALIZE_STD,
                   load_config_file)


class SkySegmentationDeeplab:
    """Class for performing sky segmentation using the DeepLab model."""
    def __init__(self, weights_path, config_file: str = None):
        """Initializes an instance of the SkySegmentationDeeplab class.

        Args:
            weights_path (str): The path to the trained model weights file.
            config_file (str): The path to the configuration file in YAML format.

        """
        config = load_config_file(config_file=config_file)

        model_name = config.get("model_name", DEFAULT_MODEL_NAME)
        pretrained_weights = config.get("pretrained_weights",
                                             DEFAULT_PRETRAINED_WEIGHTS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.segmenter = SegmentationModel(
            model_name=model_name,
            pretrained_weights=pretrained_weights,
            device=self.device
        )
        self.segmenter.load(weights_path)
        self.segmenter.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((DEFAULT_IMG_HEIGHT, DEFAULT_IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=NORMALIZE_MEAN,
                std=NORMALIZE_STD)
        ])


    def predict(self, image_path:str) -> Image:
        """Returns a PIL Image of the predicted sky mask for the input image.

        Args:
            image_path (str): The path to the input image.

        Returns:
            Image: A PIL Image of the predicted sky mask.

        """
        # Load and transform the input image
        input_image = Image.open(image_path)
        input_tensor = self.transform(input_image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            mask = torch.sigmoid(self.segmenter(input_tensor))

        mask = (mask > 0.5).float().cpu().numpy()
        # Convert from ndarray to PIL Image
        mask = Image.fromarray((mask.squeeze(0).squeeze(0) * 255).astype(np.uint8))
        # Resize to original size
        return mask.resize(input_image.size)


class SkySegmentationGrabCut:
    """Class for performing sky segmentation using the GrabCut method."""
    def __init__(self, config_file: str = None) -> None:
        """
        Initializes an instance of the SkySegmentationGrabCut class.

        Args:
            config_file (str): The path to the configuration file in YAML format.

        """
        config = load_config_file(config_file=config_file)
        self.segmenter = SegmentationGrabCut(config=config)

    def predict(self, image_path:str) -> np.ndarray:
        """Returns a numpy array of the predicted sky mask for the input image.

        Args:
            image_path (str): The path to the input image.

        Returns:
        """
        image = cv2.imread(image_path)
        mask = self.segmenter.segment(image)
        return cv2.resize(mask, image.shape[:-1][::-1])
