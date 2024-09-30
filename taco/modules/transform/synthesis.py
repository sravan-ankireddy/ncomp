import torch
import torch.nn as nn
from compressai.layers import AttentionBlock
from modules.layers.conv import deconv
from modules.layers.res_blk import ResidualBottleneck

# injector, extractor
# class Injector(nn.Module) : 
#     def __init__(self, dim, text_feature_dim, num_attn_head=8) :
#         super().__init__()

#         self.text_feature_dim_proj = nn.Linear(text_feature_dim, dim)
#         nn.init.kaiming_normal_(self.text_feature_dim_proj.weight)

#         self.image_norm = nn.LayerNorm(dim)
#         self.text_norm = nn.LayerNorm(dim)

#         self.cross_attn = nn.MultiheadAttention(dim, num_attn_head, batch_first=True)
#         self.gamma = nn.Parameter(0.0 * torch.ones((dim)), requires_grad=True) 
    
#     def forward(self, image_features:torch.Tensor, text_features:torch.Tensor) :    
        
#         b,c,h,w = image_features.size()
#         image_features = image_features.contiguous().flatten(2).permute(0,2,1)

#         text_features = self.text_feature_dim_proj(text_features)

#         text_features = self.text_norm(text_features)
#         image_features = self.image_norm(image_features)                                                                                                                                                                          

#         attn_out, attn_weights = self.cross_attn(image_features, text_features, text_features)
        
#         image_features = image_features + self.gamma * attn_out
#         image_features = image_features.contiguous().permute(0,2,1).view(b,c,h,w)

#         return image_features

# class Extractor(nn.Module) :
#     def __init__(self, dim, text_feature_dim, num_attn_head=8) :
#         super().__init__()

#         self.image_feature_dim_proj = nn.Linear(dim, text_feature_dim)
#         nn.init.kaiming_normal_(self.image_feature_dim_proj.weight)

#         self.image_norm = nn.LayerNorm(text_feature_dim)
#         self.text_norm = nn.LayerNorm(text_feature_dim)
#         self.cross_attn = nn.MultiheadAttention(text_feature_dim, num_attn_head, batch_first=True)
    
#     def forward(self, text_features:torch.Tensor, image_features:torch.Tensor) :

#         image_features = image_features.contiguous().flatten(2).permute(0,2,1)

#         image_features = self.image_feature_dim_proj(image_features)

#         text_features = self.text_norm(text_features)
#         image_features = self.image_norm(image_features)

#         attn_out, attn_weights = self.cross_attn(text_features, image_features, image_features)
        
#         text_features = text_features + attn_out

#         return text_features
    
## v3

class Injector(nn.Module):
    def __init__(self, dim, text_feature_dim, num_attn_head=8):
        super().__init__()

        # Project text features to match the image feature dimension
        self.text_feature_dim_proj = nn.Linear(text_feature_dim, dim)
        nn.init.kaiming_normal_(self.text_feature_dim_proj.weight)

        # Layer norms for image and text features
        self.image_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)

        # Cross-attention where Q is image, K and V are text
        self.cross_attn_image_text = nn.MultiheadAttention(dim, num_attn_head, batch_first=True)

        # Cross-attention where Q is image, K and V are image + text
        self.cross_attn_image_both = nn.MultiheadAttention(dim, num_attn_head, batch_first=True)

        # Parameters for weighted residual connections
        self.gamma1 = nn.Parameter(0.0 * torch.ones((dim)), requires_grad=True)
        self.gamma2 = nn.Parameter(0.0 * torch.ones((dim)), requires_grad=True)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
        # Get the batch size, channels, height, and width of image features
        b, c, h, w = image_features.size()

        # Flatten and permute image features to prepare for attention (batch_size, num_patches, channels)
        image_features = image_features.contiguous().flatten(2).permute(0, 2, 1)

        # Project and normalize text features
        text_features = self.text_feature_dim_proj(text_features)
        text_features = self.text_norm(text_features)

        # Normalize image features
        image_features = self.image_norm(image_features)

        ### First Cross-Attention (Image attending to Text)
        attn_out_image_text, attn_weights_image_text = self.cross_attn_image_text(
            image_features, text_features, text_features
        )

        ### Second Cross-Attention (Image attending to Image + Text)
        combined_kv_features = torch.cat([image_features, text_features], dim=1)
        attn_out_image_both, attn_weights_image_both = self.cross_attn_image_both(
            image_features, combined_kv_features, combined_kv_features
        )

        ### Aggregate the results of both cross-attention mechanisms
        # You can change the aggregation method (sum, mean, etc.)
        image_features = image_features + self.gamma1 * attn_out_image_text + self.gamma2 * attn_out_image_both

        # Reshape back to original image dimensions (batch_size, channels, height, width)
        image_features = image_features.contiguous().permute(0, 2, 1).view(b, c, h, w)

        return image_features



class Extractor(nn.Module):
    def __init__(self, dim, text_feature_dim, num_attn_head=8):
        super().__init__()

        # Project image features to match the text feature dimension
        self.image_feature_dim_proj = nn.Linear(dim, text_feature_dim)
        nn.init.kaiming_normal_(self.image_feature_dim_proj.weight)

        # Layer norms for image and text features
        self.image_norm = nn.LayerNorm(text_feature_dim)
        self.text_norm = nn.LayerNorm(text_feature_dim)

        # Cross-attention where Q is text, K and V are image
        self.cross_attn_text_image = nn.MultiheadAttention(text_feature_dim, num_attn_head, batch_first=True)

        # Cross-attention where Q is text, K and V are text + image
        self.cross_attn_text_both = nn.MultiheadAttention(text_feature_dim, num_attn_head, batch_first=True)

        # Parameters for weighted residual connections
        self.gamma1 = nn.Parameter(0.0 * torch.ones((text_feature_dim)), requires_grad=True)
        self.gamma2 = nn.Parameter(0.0 * torch.ones((text_feature_dim)), requires_grad=True)

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor):
        # Flatten and permute image features to prepare for attention (batch_size, num_patches, channels)
        image_features = image_features.contiguous().flatten(2).permute(0, 2, 1)

        # Project and normalize image features
        image_features = self.image_feature_dim_proj(image_features)
        image_features = self.image_norm(image_features)

        # Normalize text features
        text_features = self.text_norm(text_features)

        ### First Cross-Attention (Text attending to Image)
        attn_out_text_image, attn_weights_text_image = self.cross_attn_text_image(
            text_features, image_features, image_features
        )

        ### Second Cross-Attention (Text attending to Text + Image)
        combined_kv_features = torch.cat([text_features, image_features], dim=1)
        attn_out_text_both, attn_weights_text_both = self.cross_attn_text_both(
            text_features, combined_kv_features, combined_kv_features
        )

        ### Aggregate the results of both cross-attention mechanisms
        text_features = text_features + self.gamma1 * attn_out_text_image + self.gamma2 * attn_out_text_both

        return text_features


class SynthesisTransformEX_text(nn.Module):
    def __init__(self, N, M, text_embedding_dim, act=nn.ReLU) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            # injector 들어갈 구간
            Injector(N, text_embedding_dim),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            # extractor 들어갈 구간
            Extractor(N, text_embedding_dim),
            # injector 들어갈 구간
            Injector(N, text_embedding_dim),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, 3)
        )

    def forward(self, x,text_embeddings):
        for layer in self.synthesis_transform:
            if type(layer) == Injector :
                x = layer(x, text_embeddings)
            elif type(layer) == Extractor :
                text_embeddings = layer(text_embeddings, x)
            else:
                x = layer(x)
        return x
    
class SynthesisTransformEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            # injector 들어갈 구간
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            # extractor 들어갈 구간
            # injector 들어갈 구간
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, 3)
        )

    def forward(self, x):
        x = self.synthesis_transform(x)
        return x


class HyperSynthesisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            deconv(N, M),
            act(),
            deconv(M, M * 3 // 2),
            act(),
            deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.increase(x)
        return x
