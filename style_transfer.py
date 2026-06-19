# 1. setup and initialize model 
import torch # the core library for building/training machine learning model
from torchvision import transforms, models # image transformations and provides pre-trained models

# 2. Load and preprocess Images 
import requests # Library for making HTTP requests
from PIL import Image # Python Imaging Library (opening, manipulating, saving images)
from io import BytesIO # Provides a way to handle binary data in memory (image data from web requests)

# 3. Extract content and style features and compute Gram Matrix

# 4. Run style transfer and optimize generated image
import torch.optim as optim # module for optimization algorithms like Adam


class ModelSetup:

    def __init__(self, model_name = 'vgg19', device = None):
        """
        Initializing the model and setting up the device
        :param model_name: The name of the model to load from PyTorch
        :param_device: The computation device ('cuda' or 'cpu')
        """

        # checks if CUDA-compatible GPU is avaiable
        # CUDA is a parallel computing platform and API by NVIDIA, allows NVIDIA GPU's to accelerate general-purpose tasks(deep learning)
        # otherwise use CPU
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_name)

        # Freeze VGG weights for no gradient updates
        # We only optimize image's pixels not the model itself
        # It moves the VGG model to the device( GPU or CPU) for computation 
        # example, parameters(weights, biases) and computations are transferred from RAM to GPU's memory (VRAM) for faster computation 
        self.freeze_parameters()

        self.model.to(self.device)

    # use features part of VGG19(do not need the classification part, Fully connected layer)
    def load_model(self, model_name):
        if model_name == 'vgg19':
            return models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    # Tells PyTorch not to compute gradients of VGG19's parameters when we pass data into the model 
    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def get_model(self):
        return self.model 
    
    def get_device(self):
        return self.device


class ImageProcessor: 

    # initalize with max_size we set transform_pipeline to None(do dont know the image yet)
    def __init__(self, max_size = 400, target_size=None):
        self.max_size = max_size
        self.target_size = target_size
        self.transform_pipeline = None

    # If image is url, we download the image with Requests
    # Then prepare donwloaded image for use in the model (content attribute contains raw binary)
    # Then convert image into RGB
    def load_raw_image(self, img_path):
        if "http" in img_path:
            # server sends the file as binary data stored in response.content
            response = requests.get(img_path)

            # BytesIO(response.content): wraps raw bytes into a file-like object in memory
            # Image.open(): open the byte stream
            # .convert('RGB'): Converts the image to RGB color mode since image could be in another format like grayscale
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')
        return image

    
    def create_transform_pipeline(self, image):
        if max(image.size) > self.max_size:
            size = self.max_size
        else:
            size = max(image.size)
        if self.target_size is not None:
            size = self.target_size

        # Resize: ensure content and style images are the same size
        # ToTensor: convert PIL image to PyTorch Tensor (C,H,N) with values in [0,1]
        # Normalize: Adjusts channels using Image Net Mean and STD
        self.transform_pipeline = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    # Batch Dim: Adds extra dimension (1,C,H,W) with .unsqueeze(0)
    def transform_image(self, image):
        if self.transform_pipeline is None:
            self.create_transform_pipeline(image)

        # Remove unused alpha (transparency) channel (the :3) and keep RGB
        # At this point the image 3D tensor: (rgb_channels, height, width) for example (3, 128, 128) - where:
            # image = torch.tensor([
            # [[value_1, value_2, ..., value_256], ..., [value_1, ..., value_256]],     # Red channel (128 rows, 256 values each)  Note:[value_1, ..., value_256] is one row even though looks like column
            # [[value_1, value_2, ..., value_256], ..., [value_1, ..., value_256]],     # Green channel (128 rows, 256 values each)
            # [[value_1, value_2, ..., value_256], ..., [value_1, ..., value_256]],     # Blue channel (128 rows, 256 values each)
            # )  
            # 128 rows in each channel (for the height of img)
            # 256 values in each row array (for the width of img)

        # Unsqueeze(0) adds a new dimension - the batch dimension - representing no. of images in the batch (will just be 1 in our case but useful feature to have)
        # Neural networks are designed to process batches of data because it’s faster and more efficient
        # Now the 4D tensor: (batch size, rgb_channels, height, width)
        # Example Tensor: (2, 3, 128, 128) - only 1 batch dimension in our code but for understanding lets see 2:
            # batch_images = [
            #   [
            #       Image 1
            #       [[R1, R2, ..., R256], ..., [R1, ..., R256]],  # Red channel
            #       [[G1, G2, ..., G256], ..., [G1, ..., G256]],  # Green channel
            #       [[B1, B2, ..., B256], ..., [B1, ..., B256]]   # Blue channel
            #   ],
            #   [
            #       Image 2
            #       [[R1, R2, ..., R256], ..., [R1, ..., R256]],  # Red channel
            #       [[G1, G2, ..., G256], ..., [G1, ..., G256]],  # Green channel
            #       [[B1, B2, ..., B256], ..., [B1, ..., B256]]   # Blue channel
            #   ]
            #]
        # Example at position (batch=0, channel=1, height=30, width=128) = tensor[0, 1, 30, 128] = a normalized value like 0.456 (Green channel) 

        transformed_image = self.transform_pipeline(image)[:3,:,:].unsqueeze(0)
        return transformed_image

class ImageVisualizer:
    pass

class FeatureExtractor:
    pass

class FeatureManager:
    pass

class StyleTransfer:
    pass


# Initialize variables 
model_setup = ModelSetup()
vgg = model_setup.get_model()
device = model_setup.get_device()
processor = ImageProcessor()
visualizer = ImageVisualizer(save_path="./results")
feature_extractor = FeatureExtractor(model=vgg)
feature_manager = FeatureManager(feature_extractor=feature_extractor)

# Prepare Images
content_raw = processor.load_raw_image('./content.png')
style_raw = processor.load_raw_image('./style.jpg')

# Preprocess the images and move to the device
content = processor.transform_image(content_raw).to(device)
style = processor.transform_image(style_raw).to(device)

# Display images side by side
visualizer.save_images_side_by_side()
visualizer.display_images_side_by_side()

# Get content and style features and style gram matrix
feature_manager.compute_features(content_image=content, style_image=style)
feature_manager.compute_style_gram_matrix()

# Initialise Style Transfer
style_transfer = StyleTransfer()

# Generate Image and Optimise 
style_transfer.set_generated_image(content_image=content)
final_image = style_transfer.optimise()


