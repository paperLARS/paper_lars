import timm
import torch
from torch import nn

class ClassToken(nn.Module):
    """
    Creates a learnable classification token to be prepended to the sequence.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x):
        batch_size = x.size(0)
        return self.cls_token.expand(batch_size, -1, -1)

def mlp(in_features, hidden_features=None, out_features=None, dropout_rate=0.1):
    """
    A simple MLP block with two linear layers and a GELU activation.
    """
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_features, out_features),
        nn.Dropout(dropout_rate)
    )

class PatchEmbedding(nn.Module):
    """
    Splits images into patches and projects them into a higher-dimensional space.
    """
    def __init__(self, image_size, patch_size, in_channels, hidden_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x + self.pos_embedding

class TransformerEncoder(nn.Module):
    """
    Implements a single Transformer encoder layer.
    """
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = mlp(hidden_dim, mlp_dim, hidden_dim, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        skip_1 = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = skip_1 + self.dropout1(attn_output)

        skip_2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = skip_2 + self.dropout2(x)

        return x

class ViT(nn.Module):
    """
    Assembles the complete Vision Transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.patch_embedding = PatchEmbedding(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            in_channels=config["num_channels"],
            hidden_dim=config["hidden_dim"]
        )
        
        self.cls_token = ClassToken(config["hidden_dim"])

        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(
                hidden_dim=config["hidden_dim"],
                num_heads=config["num_heads"],
                mlp_dim=config["mlp_dim"],
                dropout_rate=config["dropout_rate"]
            ) for _ in range(config["num_layers"])
        ])

        self.norm = nn.LayerNorm(config["hidden_dim"])
        self.head = nn.Linear(config["hidden_dim"], 1)
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_token = self.cls_token(x)
        x = torch.cat((cls_token, x), dim=1)

        for encoder in self.transformer_encoders:
            x = encoder(x)
        
        x = self.norm(x)
        cls_token_output = x[:, 0]
        output = self.head(self.dropout(cls_token_output))
        
        return output
        
def create_pretrained_vit(config, model_name="vit_base_patch16_224"):
    """
    Loads a pre-trained Vision Transformer model from the timm library,
    freezes its parameters, and replaces the classification head.
    """
    model = timm.create_model(model_name, pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 1) 
    return model
