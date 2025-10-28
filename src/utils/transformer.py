import logging
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_schedule_fn

# Configure logging
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def linear_schedule(start_lr = 3e-4, end_lr = 1e-5, total_timesteps=1e6):
    lr_schedule = get_schedule_fn(1e-4)  # For constant LR
    # OR for decaying LR:
    lr_schedule = lambda progress: start_lr - (start_lr - end_lr) * progress
    return lr_schedule


class TimeSeriesTransformer(nn.Module):
    """
    A Transformer-based model for time series data.
    This class projects input features to an embedding, adds positional
    encodings, and then processes the inputs using a Transformer encoder.
    Finally, a decoder layer is used to produce the output.
    Args:
        input_size (int): Number of features in the input time series data.
        embed_dim (int): Dimensionality of the learned embedding space.
        num_heads (int): Number of attention heads in each Transformer layer.
        num_layers (int): Number of Transformer encoder layers.
        sequence_length (int): Length of the input sequences (time steps).
        dropout (float, optional): Dropout probability to apply in the
            Transformer encoder layers. Defaults to 0.1.
    Attributes:
        model_type (str): Identifier for the model type ('Transformer').
        embedding (nn.Linear): Linear layer for input feature embedding.
        positional_encoding (torch.nn.Parameter): Parameter storing the
            positional encodings used to retain temporal information.
        transformer_encoder (nn.TransformerEncoder): Stack of Transformer
            encoder layers with optional final LayerNorm.
        decoder (nn.Linear): Linear layer used to produce the final output
            dimensions.
    Forward Inputs:
        src (torch.Tensor): Input tensor of shape (batch_size, sequence_length,
            input_size).
    Forward Returns:
        torch.Tensor: Output tensor of shape (batch_size, embed_dim) from the
            last time step.
    Raises:
        ValueError: If the model output contains NaN or Inf values, indicating
            numerical instability.
    """
    # input_size: Input features အရေအတွက် (ဥပမာ 10၊ price + SMA/RSI indicators စတာ)။
    # embed_dim: Internal embedding အတိုင်းအတာ (ဥပမာ 64၊ data ကို ပိုနက်ရှိုင်း အောင် ပြောင်း)။
    # num_heads: Attention heads အရေအတွက် (multi-head attention အတွက်၊ မတူညီ အနေနဲ့ အာရုံ စိုက်)။
    # num_layers: Encoder layers အရေအတွက် (ဥပမာ 2၊ ရိုးရှင်း ထားတာ)။
    # sequence_length: Input sequence အရှည် (ဥပမာ 20 timesteps)။
    # dropout=0.1: Overfitting ကနေ ကာကွယ် တဲ့ dropout rate။
    def __init__(self, input_size, embed_dim, num_heads, num_layers,sequence_length, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.embed_dim = embed_dim

        # Embedding layer to project input features to embed_dim dimensions
        self.embedding = nn.Linear(input_size, embed_dim).to(device)

        # Positional encoding parameter
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, embed_dim).to(device))

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            norm_first=True  # Apply LayerNorm before attention and feedforward
        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim).to(device) # Add LayerNorm at the end of the encoder
        )

        # Decoder layer to produce final output
        self.decoder = nn.Linear(embed_dim, embed_dim).to(device)

    def forward(self, src):
        # Apply embedding layer and add positional encoding
        src = self.embedding(src) + self.positional_encoding

        # Pass through the transformer encoder
        output = self.transformer_encoder(src)

        # Pass through the decoder layer
        output = self.decoder(output)

        # Check for NaN or Inf values for debugging
        if torch.isnan(output).any() or torch.isinf(output).any():
            logger.error("Transformer output contains NaN or Inf values")
            raise ValueError("Transformer output contains NaN or Inf values")

        # Return the output from the last time step
        return output[:, -1, :]
    
    
class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that normalizes input observations and processes them
    using a transformer-based architecture for dimensionality reduction and enhanced
    feature representation.
    Parameters:
        observation_space (gym.spaces.Box): Defines the shape and limits of input data.
        sequence_length (int): The length of the time series to be processed.
    Attributes:
        layernorm_before (nn.LayerNorm): Normalizes input data to improve training stability.
        transformer (TimeSeriesTransformer): Processes normalized input sequences and extracts features.
    Methods:
        forward(observations):
            Applies layer normalization to the incoming observations, then passes them
            through the transformer. Raises a ValueError if invalid values (NaNs or inf)
            are detected in the output.
    """
    
    def __init__(self, observation_space: gym.spaces.Box, sequence_length):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=64)
        num_features = observation_space.shape[1]  # Should be 10 in this case

        # Ensure that embed_dim is divisible by num_heads
        embed_dim = 64
        num_heads = 2

        self.layernorm_before = nn.LayerNorm(num_features) # Added Layer Normalization before transformer

        self.transformer = TimeSeriesTransformer(
            input_size=num_features,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=2,
            sequence_length =sequence_length
        )

    def forward(self, observations):
        # မူရင်း input tensor ရဲ့ device ကို မှတ်သားထားပါ
        input_device = observations.device
        
        # Apply layer normalization
        # Apply layer normalization, ဝင်လာတဲ့ observations ကို Transformer ရဲ့ device ပေါ်ကို ရွှေ့ပါ
        normalized_observations = self.layernorm_before(observations.float().to(device)) # Ensure float type

        x = self.transformer(normalized_observations)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("Invalid values in transformer output")
            raise ValueError("Invalid values in transformer output")
        
        # ⚠️ ပြင်ဆင်ချက်: Output tensor ကို မူရင်း input tensor ရဲ့ device သို့ ပြန်ပို့ပါ
        # PPO Agent ရဲ့ Policy/Value Network က အလုပ်လုပ်တဲ့ device ပေါ်ကို ပြန်ပို့ဖို့ လိုပါတယ်။
        # သို့သော်လည်း၊ Stable-Baselines3 က Policy/Value Network ကို နောက်ပိုင်းမှာ to(device) နဲ့ ရွှေ့တဲ့အတွက်
        # ဒီနေရာမှာ အန္တရာယ်ကင်းအောင် မူရင်း input device ကို ပြန်ပို့တာ ဒါမှမဟုတ် Agent သုံးမယ့် device ပေါ်မှာပဲ ထားတာ နှစ်မျိုး လုပ်နိုင်ပါတယ်။
        # အကောင်းဆုံးကတော့ Policy Network တွေက GPU ပေါ်မှာရှိရင် GPU မှာပဲ ထားခဲ့တာပါ။
        
        # သို့သော်လည်း၊ SB3 ရဲ့ စံနှုန်းကို လိုက်နာဖို့၊ CPU ပေါ်ကလာရင် CPU ကို ပြန်ပို့တာ ပိုကောင်းပါတယ်။
        if str(input_device) == 'cpu':
            return x.to(input_device)
        else:
             # Agent က GPU မှာ Run ရင်တော့ GPU မှာပဲ ထားခဲ့ပါ
            return x
