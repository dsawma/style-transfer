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
import tqdm import tqdm # Progress bar 

# 5. Image Visualizer
import matplotlib.pyplot as plt # Part of Matplotlib(plotting and visualizing data or images)
import numpy as np # Numerical library (handling arrays, math operations, matrix manipulation)



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
    def __init__(self, save_path =None):
        self.save_path = save_path
    
    def display_image(self, image_tensor, title= None):
        image = self.tensor_to_image (image_tensor)
        plt.imshow(image)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()
    
    def display_images_side_by_side(self,images, titles=None):
        num_images = len(images)
        fig,axes = plt.subplots(1, num_images, figsize = (20,10))
        if num_images == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            image = self.tensor_to_image(images[i])
            ax.imshow(image)
            ax.axis('off')
            if titles and i < len(titles):
                ax.set_title(titles[i])
        plt.show()

    def save_image(self, image_tensor, filename):
        """Save a tensor as a image to the path"""
        if not self.save_path:
            raise ValueError("Save path not specified")
        image = self.tensor_to_image(image_tensor)
        save_path = f"{self.save_path}/{filename}"
        plt.imsave(save_path, image)
        print(f"Image saved to {save_path}")

    def save_images_side_by_side(self, images, filename, titles = None):
        """Save multiple images side by side as a single image"""
        if not self.save_path:
            raise ValueError("Save path not specified")
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=(20,10))
        if num_images == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            image = self.tensor_to_image(images[i])
            ax.imshow(image)
            ax.axis('off')
            if titles and i < len(titles):
                ax.set_title(titles[i])
        
        save_path = f"{self.save_path}/{filename}"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Side-by-side image saved to {save_path}")


    # helper function for un-normalizing a tesnor and converting it to Numpy image for display
    # Move tensor to CPU, detach gradients, and convert to NumPy( Processing and Visualization must happen on CPU)
    # Detaching from gradient computations also improves memory usage and speeds up working the image
    @staticmethod
    def tensor_to_image(tensor):
        image = tensor.to("cpu").clone().detech()
        image = image.numpy().squeeze() # Remove batch dimension 
        image = image.transpose(1,2,0) # Rearrange to (height, width, channels) - expected by libraries like NumPy and matplotlib
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406)) # Un-normalize
        image = image.clip(0,1) # clip pixel values to [0,1] range 
        return image
    
# Acts as a controller delegating the computations to the feature extractor class 
# Responsible for getting the features 
# Based on Gatys et al. (2016) for style and content extraction 
# Lower layers for basic features like details and structure(conv1_1 = edges, textures)
# Deeper layers for higher-level features like patterns adn spatial awareness (conv5_1 = object structure)

class FeatureExtractor:
    def __init__(self, model):
        self.model = model 

        # the keys are the indices of layers in vgg19.features module(container of layers)
        # '21' is used to extract content features, rest are used for style features
        # ex. Layer Name 0: FIrst convolutional layer in block 1 (conv1_1) = Layer name: 0, Layer: Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.layers = {
            '0':'conv1_1',
            '5':'conv2_1',
            '10':'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2', # content representation 
            '28' : 'conv5_1'
        }
    
    def get_features(self,image):
        """
        Extracts features from layers of the model for a given image
        :param image: Input tensor of shape(batch_size, normalized_rgb_channels, height, width) 
        :return: Dictionary of features for the specified layers 
        """

        features = {} # Dict to store extracted features for specified layers

        # model._modules.item(): dict of model's layers(key: 'name' is layer name(index), value: 'layer' is layer object)
        # ex. Layer name: 5, Layer: Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # The image tensor is passed through each of the layers of the model, the layer is applied to the image to get feature map 
        for name, layer in self.model._modules.items():
            image = layer(image)
            if name in self.layers:
                features[self.layers[name]] = image
        return features
    
    @staticmethod
    def gram_matrix(tensor):
        """
        Computes the Gram matrix for style representation 
        :param tensor: Input tesnor of shape (batch_size, channels, height, width)
        :return: Gram matrix
        """
        b, d, h, w= tensor.size() # batch_size not needed
        # Reshape tensor to (channels, height * width) 
        # ex. feature map tensor of (64 * H * W) 64 filters. Will reshape to 64(H * W) 
        # Where each row corresponds to a filter's activations flattened across all spacial locations
        tensor = tensor.view(b*d, h*w)
        gram = torch.mm(tensor, tensor.t())
        return gram 

# Controller, delegating the computation to the Feature Extraction class
class FeatureManager:
    def __init__ (self, feature_extractor):
        """
        Initialize with a feature extractor
        :param feature_extractor: An instance of FeatureExtractor for extracting features and Gram matrices
        """
        self.feature_extractor = feature_extractor
        self.content_feature_set = None
        self.style_feature_set = None
        self.style_grams = None

    def compute_features(self, content_image, style_image):
        """
        Extract features from content and style images with .get_features function 
        :param content_image: Preprocessed content image tensor
        :param style_image: Preprocesssed style image tensor
        """
        self.content_feature_set = self.feature_extractor.get_features(content_image)
        self.style_feature_set = self.feature_extractor.get_features(style_image)

        # View the extracted features
        aggregated = self.style_feature_set['conv1_1'][0].mean(dim=0)  # Average across channels
        plt.imshow(aggregated.cpu().detach().numpy(), cmap='viridis')
        plt.show()


    # Computes the Gram Matrix with .gram_matrix function for the style image for each layer
    # for each layer in style_feature_set: go through the style image's layer-feature map dict
    # key = layer name(ex. 'conv1_1') and value = feature map (3D tesnor with (channels, height, width))
    # style_feature_set[layer] is used to access the feature map 
    # The number of filters in a layer is the number of feature channels in the output feature map 
    # Each filter produces one feature channel(2D matrix of shape (height, width))
    # ex. If a layer has 64 filters, the output feature map will have 64 channels
    # {layer: ...}: creates a dict (Key = layer_name (e.g. 'conv1_1') , Value = Gram matrix(2D tensor- (feature_channels, feature_channels))
    def compute_style_gram_matrix(self):
        self.style_grams = {
            layer: self.feature_extractor.gram_matix(self.style_feature_set[layer])
            for layer in self.style_feature_set
        }

    def get_content_feature_set(self):
        return self.content_feature_set
    
    def get_style_grams(self):
        return self.style_grams
    
class StyleTransfer:
    def __init__(self, content_feature_set, style_grams, feature_extractor, visualizer, device, style_weights, content_img, style_img, content_weight=1, style_weight=1e6):
        self.content_feature_set = content_feature_set
        self.style_grams = style_grams
        self.feature_extractor = feature_extractor
        self.visualizer = visualizer
        self.device = device
        self.style_weights = style_weights
        self.content_img = content_img
        self.style_img = style_img
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.generated_image = None

    # Take original image and adjust pixels that aligns best with content and style
    # Treate entire image tensor as a trainable object where each individual pixel value acts as parameter that PyTorch 
    # will update to minimize the content and style losses
    def set_generated_image(self, content_image):
        """
        Initialize the generated image as a copy of the content image
        :param content_image: Preprocessed content image tensor
        """
        self.generated_image = content_image.clone().requires_grad_(True).to(self.device)


    # Returns both content and style loss
    # 1.Content loss:
    # Compares feature maps of generated image to the content image feature maps we extracted (compare current state of our model's output to the target)
    # We do this on conv4_2 since its the representation of the features we want to return without too much influence from low-level
    # Then Calculate the Mean Square Error
    # 2. Style loss:
    # Initialize scalar(represents generated img deviation from style img) style loss at 0 to 
    # We go through all pre-selected layers, we extracted the feature maps
    # Get feature map of generated target image for current layer and compare its Gram Matrix
    # (The Gram matrix encodes correlations between all pairs of feature maps)
    def compute_losses(self, target_feature_set):
        """
        Compute content and style losses
        :param target_feature_set: Extracted features of the generated image
        :return: Tuple (content_loss, style_loss)
        """
        content_loss = torch.mean((target_feature_set['conv4_2'] - self.content_feature_set['conv4_2']) ** 2)
        style_loss = 0

        # Style loss:
        # 1. Get the feature_maps of generated target img for current layerand compute its Gram matrix
        # 2. Retrieve precomputed Gram matrix of style imaege for the same layer
        # 3. Calculate the MSE loss between the target Gram matrix and the style image Gram matrix
        # 4. Weight the loss using the predefined style weigths for the layer 
        # 5. Normalize the lsos by the total number of elements in the feature map (ensures loss for layer is independent of feature map size)
        # 6. Accumulate the style loss across all specified layers

        for layer in self.style_weights:
            target_feature = target_feature_set[layer]
            target_gram = self.feature_extractor.gram_matrix(target_feature)
            _, d, h, w = target_feature.shape

            style_gram = self.style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
            weighted_layer_style_loss = self.style_weights[layer] * layer_style_loss
            style_loss += weighted_layer_style_loss / (d * h * w)
        return content_loss, style_loss

    def optimise(self, steps=5000, save_every=250):
        """
        Optimize the generated image to minimize content and style losses
        :param steps: Number of optimization steps
        :param save_every: Interva; for displaying intermediate results
        :return: Final stylized image tensor
        """
        optimizer = optim.Adam([self.generated_image], lr=0.003)
        with tqdm(total=steps, desc="Optimizing Style Transfer") as pbar:
            for step_number in range(1, steps + 1):

                # Extract features from the generated image
                target_feature_set = self.feature_extractor.get_features(self.generated_image)

                # Compute aggregate loss for both content and style
                content_loss, style_loss = self.compute_losses(target_feature_set)

                total_loss = self.content_weight * content_loss + self.style_weight * style_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                pbar.set_postfix({'Total Loss': f"{total_loss.item():.4f}"}, refresh=True)
                pbar.update(1)

                if step_number % save_every ==0:
                    print(f"Step {step_number}, Total Loss: {total_loss.item():.4f}")
                    self.visualizer.save_image(self.generated_image, filename= f"step_{step_number}.png")
                    self.visualizer.display_image(self.generated_image, title=f"Step {step_number}")

            self.visualizer.save_images_side_by_side(
                images=[self.content_img, self.style_img, self.generated_image],
                filename = "final_side_by_side.png",
                titles = ["Content Image", "Style Image", "Stylized Image"]
            )
        return self.generated_image


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
visualizer.save_images_side_by_side(images=[content, style], filename="side_by_side.png", titles=["Content Image", "Style Image"])
visualizer.display_images_side_by_side(images=[content, style], titles=["Content Image", "Style Image"])

# Get content and style features and style gram matrix
feature_manager.compute_features(content_image=content, style_image=style)
feature_manager.compute_style_gram_matrix()

# Initialise Style Transfer
style_transfer = StyleTransfer()

# Generate Image and Optimise 
style_transfer.set_generated_image(content_image=content)
final_image = style_transfer.optimise()


