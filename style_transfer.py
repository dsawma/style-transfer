# 1. setup and initialize model 
import torch # the core library for building/training machine learning model
from torchvision import transforms, models # image transformations and provides pre-trained models

# 2. Load and preprocess Images 

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
    def __init__(self):
        pass

    def load_raw_image(self, img_path):
        pass
    def create_transform_pipeline(self,image):
        pass

    def transform_image(self, image):
        pass

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


