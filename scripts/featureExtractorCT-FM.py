# Imports
import torch
from lighter_zoo import SegResEncoder
from monai.transforms import (
    Compose, LoadImage, EnsureType, Orientation,
    ScaleIntensityRange, CropForeground
)
from monai.inferers import SlidingWindowInferer

# Load pre-trained model
model = SegResEncoder.from_pretrained(
    "project-lighter/ct_fm_feature_extractor"
)
model.eval()

# Preprocessing pipeline
preprocess = Compose([
    LoadImage(ensure_channel_first=True),  # Load image and ensure channel dimension
    EnsureType(),                         # Ensure correct data type
    Orientation(axcodes="SPL"),           # Standardize orientation
    # Scale intensity to [0,1] range, clipping outliers
    ScaleIntensityRange(
        a_min=-1024,    # Min HU value
        a_max=2048,     # Max HU value
        b_min=0,        # Target min
        b_max=1,        # Target max
        clip=True       # Clip values outside range
    ),
    CropForeground()    # Remove background to reduce computation
])

# Input path
input_path =  r"C:\Users\danny\Documents\Code\Imperial_Dissertation\data\AeroPath\1\1_CT_HR.nii.gz"

# Preprocess input
input_tensor = preprocess(input_path)

# Run inference
with torch.no_grad():
    output = model(input_tensor.unsqueeze(0))[-1]

    # Average pooling compressed the feature vector across all patches. If this is not desired, remove this line and 
    # use the output tensor directly which will give you the feature maps in a low-dimensional space.
    avg_output = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()

print("✅ Feature extraction completed")
print(f"Output shape: {avg_output.shape}")

# ✅ Feature extraction completed
# Output shape: torch.Size([512])

# Plot distribution of features
import matplotlib.pyplot as plt
_ = plt.hist(avg_output.cpu().numpy(), bins=100)                                                        
