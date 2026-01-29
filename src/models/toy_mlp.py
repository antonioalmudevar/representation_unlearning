# src/models/toy_mlp.py
import torch
from torch import nn
from . import register_model
from .utils import load_checkpoint_, set_bn_opts_


class ToyMLP(nn.Module):
    """
    Simple MLP for toy experiments.
    Projects 10-dimensional input to 2-dimensional representation space,
    then classifies into 4 classes.
    
    Architecture:
    - Input: 10-dim
    - Hidden1: 64-dim with ReLU
    - Hidden2: 32-dim with ReLU
    - Representation: 2-dim (encoder output)
    - Classifier: 2-dim -> num_classes
    """
    def __init__(
        self,
        input_dim: int = 10,
        repr_dim: int = 2,
        num_classes: int = 4,
        hidden_dims: list = None,
        dropout: float = 0.0,
        checkpoint: str = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        # Build encoder (input -> representation)
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final projection to representation space
        encoder_layers.append(nn.Linear(prev_dim, repr_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Classifier head (representation -> classes)
        self.head = nn.Linear(repr_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Load checkpoint if provided
        if checkpoint:
            load_checkpoint_(self, checkpoint)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_repr=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_repr: If True, return (logits, representation)
        
        Returns:
            logits: Class logits of shape (batch_size, num_classes)
            representation (optional): Representation of shape (batch_size, repr_dim)
        """
        repr = self.encoder(x)
        logits = self.head(repr)
        
        if return_repr:
            return logits, repr
        return logits
    
    def get_representation(self, x):
        """Extract representation without classification."""
        return self.encoder(x)


@register_model("toy_mlp")
def toy_mlp(
    input_dim: int = 10,
    repr_dim: int = 2,
    num_classes: int = 4,
    hidden_dims: list = None,
    dropout: float = 0.0,
    checkpoint: str = None,
):
    """Factory function for ToyMLP model."""
    return ToyMLP(
        input_dim=input_dim,
        repr_dim=repr_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        checkpoint=checkpoint,
    )
