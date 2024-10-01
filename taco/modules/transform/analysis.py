import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3, AttentionBlock
from modules.layers.conv import conv1x1, conv3x3, conv, deconv
from modules.layers.res_blk import *


module_ver = 3

if module_ver == 1:

    # injector, extractor
    class Injector(nn.Module) : 
        def __init__(self, dim, text_feature_dim, num_attn_head=8) :
            super().__init__()

            self.text_feature_dim_proj = nn.Linear(text_feature_dim, dim)
            nn.init.kaiming_normal_(self.text_feature_dim_proj.weight)

            self.image_norm = nn.LayerNorm(dim)
            self.text_norm = nn.LayerNorm(dim)

            self.cross_attn = nn.MultiheadAttention(dim, num_attn_head, batch_first=True)
            self.gamma = nn.Parameter(0.0 * torch.ones((dim)), requires_grad=True) 
        
        def forward(self, image_features:torch.Tensor, text_features:torch.Tensor) :    
            
            b,c,h,w = image_features.size()
            image_features = image_features.contiguous().flatten(2).permute(0,2,1)

            text_features = self.text_feature_dim_proj(text_features)

            text_features = self.text_norm(text_features)
            image_features = self.image_norm(image_features)                                                                                                                                                                          

            attn_out, attn_weights = self.cross_attn(image_features, text_features, text_features)
            
            image_features = image_features + self.gamma * attn_out
            image_features = image_features.contiguous().permute(0,2,1).view(b,c,h,w)

            return image_features

    class Extractor(nn.Module) :
        def __init__(self, dim, text_feature_dim, num_attn_head=8) :
            super().__init__()

            self.image_feature_dim_proj = nn.Linear(dim, text_feature_dim)
            nn.init.kaiming_normal_(self.image_feature_dim_proj.weight)

            self.image_norm = nn.LayerNorm(text_feature_dim)
            self.text_norm = nn.LayerNorm(text_feature_dim)
            self.cross_attn = nn.MultiheadAttention(text_feature_dim, num_attn_head, batch_first=True)
        
        def forward(self, text_features:torch.Tensor, image_features:torch.Tensor) :

            image_features = image_features.contiguous().flatten(2).permute(0,2,1)

            image_features = self.image_feature_dim_proj(image_features)

            text_features = self.text_norm(text_features)
            image_features = self.image_norm(image_features)

            attn_out, attn_weights = self.cross_attn(text_features, image_features, image_features)
            
            text_features = text_features + attn_out

            return text_features

elif module_ver == 2:

    # v2
    class Injector(nn.Module):
        def __init__(self, dim, text_feature_dim, num_heads=8):
            super().__init__()
            self.num_heads = num_heads
            self.query_proj = nn.Linear(dim, dim)
            self.key_proj_text = nn.Linear(text_feature_dim, dim)
            self.value_proj_text = nn.Linear(text_feature_dim, dim)

            # Self-attention for image features
            self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

            # Cross-attention for image features and text embeddings
            self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

            self.gamma = nn.Parameter(0.0 * torch.ones((dim)), requires_grad=True)

        def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
            # Reshape image features for attention: [batch_size, seq_len, channels]
            b, c, h, w = image_features.size()
            image_features = image_features.contiguous().view(b, c, h * w).permute(0, 2, 1)  # [batch_size, seq_len, dim]

            # Project queries from image features and keys/values from text embeddings
            query = self.query_proj(image_features)  # [batch_size, seq_len, dim]
            key_text = self.key_proj_text(text_features)  # [batch_size, text_feature_dim, dim]
            value_text = self.value_proj_text(text_features)  # [batch_size, text_feature_dim, dim]

            # Apply self-attention on the image features
            attn_image, _ = self.self_attn(query, query, query)  # [batch_size, seq_len, dim]

            # Apply cross-attention using text embeddings
            attn_text, _ = self.cross_attn(query, key_text, value_text)  # [batch_size, seq_len, dim]

            # Aggregate self-attention and cross-attention results
            image_features = attn_image + self.gamma * attn_text

            # Reshape back to original image shape: [batch_size, channels, height, width]
            image_features = image_features.permute(0, 2, 1).view(b, c, h, w)

            return image_features

    class Extractor(nn.Module):
        def __init__(self, dim, text_feature_dim, num_heads=8):
            super().__init__()
            self.num_heads = num_heads
            self.query_proj_text = nn.Linear(text_feature_dim, text_feature_dim)
            self.key_proj_image = nn.Linear(dim, text_feature_dim)
            self.value_proj_image = nn.Linear(dim, text_feature_dim)

            # Self-attention for text features
            self.self_attn_text = nn.MultiheadAttention(text_feature_dim, num_heads, batch_first=True)

            # Cross-attention for text and image features
            self.cross_attn = nn.MultiheadAttention(text_feature_dim, num_heads, batch_first=True)

            self.gamma = nn.Parameter(0.0 * torch.ones((text_feature_dim)), requires_grad=True)

        def forward(self, text_features: torch.Tensor, image_features: torch.Tensor):
            # Reshape image features for attention: [batch_size, seq_len, channels]
            b, c, h, w = image_features.size()
            image_features = image_features.contiguous().view(b, c, h * w).permute(0, 2, 1)  # [batch_size, seq_len, dim]

            # Project queries from text features and keys/values from image features
            query_text = self.query_proj_text(text_features)  # [batch_size, text_feature_dim, dim]
            key_image = self.key_proj_image(image_features)  # [batch_size, seq_len, text_feature_dim]
            value_image = self.value_proj_image(image_features)  # [batch_size, seq_len, text_feature_dim]

            # Apply self-attention on the text features
            attn_text, _ = self.self_attn_text(query_text, query_text, query_text)  # [batch_size, seq_len, text_feature_dim]

            # Apply cross-attention using image features
            attn_image, _ = self.cross_attn(query_text, key_image, value_image)  # [batch_size, seq_len, text_feature_dim]

            # Aggregate self-attention and cross-attention results
            text_features = attn_text + self.gamma * attn_image

            return text_features

elif module_ver == 3:
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


        
elif module_ver == 4:
    class Injector(nn.Module):
        def __init__(self, dim, text_feature_dim, num_attn_head=8):
            super().__init__()

            # Project text features to match the image feature dimension
            self.text_feature_dim_proj = nn.Linear(text_feature_dim, dim)
            nn.init.kaiming_normal_(self.text_feature_dim_proj.weight)

            # Layer norms for image and text features
            self.image_norm = nn.LayerNorm(dim)
            self.text_norm = nn.LayerNorm(dim)

            # Cross-attention layer where Q is image, K and V are image + text
            self.cross_attn_image_both = nn.MultiheadAttention(dim, num_attn_head, batch_first=True)

            # Parameter for weighted residual connections
            self.gamma = nn.Parameter(0.0 * torch.ones((dim)), requires_grad=True)

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

            ### Cross-Attention (Image attending to Image + Text)
            combined_kv_features = torch.cat([image_features, text_features], dim=1)
            attn_out_image_both, _ = self.cross_attn_image_both(
                image_features, combined_kv_features, combined_kv_features
            )

            ### Aggregate the results with a residual connection
            image_features = image_features + self.gamma * attn_out_image_both

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

            # Cross-attention layer where Q is text, K and V are text + image
            self.cross_attn_text_both = nn.MultiheadAttention(text_feature_dim, num_attn_head, batch_first=True)

            # Parameter for weighted residual connections
            self.gamma = nn.Parameter(0.0 * torch.ones((text_feature_dim)), requires_grad=True)

        def forward(self, text_features: torch.Tensor, image_features: torch.Tensor):
            # Flatten and permute image features to prepare for attention (batch_size, num_patches, channels)
            image_features = image_features.contiguous().flatten(2).permute(0, 2, 1)

            # Project and normalize image features
            image_features = self.image_feature_dim_proj(image_features)
            image_features = self.image_norm(image_features)

            # Normalize text features
            text_features = self.text_norm(text_features)

            ### Cross-Attention (Text attending to Text + Image)
            combined_kv_features = torch.cat([text_features, image_features], dim=1)
            attn_out_text_both, _ = self.cross_attn_text_both(
                text_features, combined_kv_features, combined_kv_features
            )

            ### Aggregate the results with a residual connection
            text_features = text_features + self.gamma * attn_out_text_both

            return text_features


class AnalysisTransformEX(nn.Module):
    def __init__(self, N, M, text_embedding_dim, act=nn.ReLU):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            conv(3, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            Injector(N, text_embedding_dim),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            Extractor(N, text_embedding_dim),
            Injector(N, text_embedding_dim),
            conv(N, M),
            AttentionBlock(M)
        )

    def forward(self, x, text_embeddings):
        
        for layer in self.analysis_transform:
            if type(layer) == Injector :
                x = layer(x, text_embeddings)
            elif type(layer) == Extractor :
                text_embeddings = layer(text_embeddings, x)
            else:
                x = layer(x)
                
        return x

class HyperAnalysisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            act(),
            conv(N, N),
            act(),
            conv(N, N)
        )

    def forward(self, x):
        x = self.reduction(x)
        return x

