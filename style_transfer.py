import torch
import torch.nn as nn 

class ModelSetup:
    def __init__(self):
        

    def load_model(self, model_name):
        if model_name == 'vgg19':
            return models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False


# Initialize variables 
model_setup = ModelSetup()
vgg = model_setup.get_model()
device = model_setup.get_device()
processor = ImageProcessor()
visualizer = ImageVisualizer(save_path="./")
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


