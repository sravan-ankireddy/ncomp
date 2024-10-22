# build DC-AE models
# full DC-AE model list: https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b
from efficientvit.ae_model_zoo import DCAE_HF

# dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f64c128-in-1.0")
# dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f32c32-in-1.0")
dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f128c512-in-1.0")

# encode
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.apps.utils.image import DMCrop

device = torch.device("cuda")
dc_ae = dc_ae.to(device).eval()

transform = transforms.Compose([
    DMCrop(512), # resolution
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
image = Image.open("assets/fig/girl.png")
x = transform(image)[None].to(device)
latent = dc_ae.encode(x)
print(latent.shape)
# breakpoint()
# decode
y = dc_ae.decode(latent)
save_image(y * 0.5 + 0.5, "demo_dc_ae.png")