import tensorflow as tf
from tensorflow.keras import layers

from model.sub_layer import MultiHeadAttention, FeedForward
from model.embed import positional_encoding


class DecoderLayer(layers.Layer):

    def __init__(self, 
                 d_h: int, 
                 head: int, 
                 d_ff: int,
                 rate: float,
                 layer_idx: int):
        """GPT decoder layer

        Args:
            d_h (int): attn hidden dim
            head (int): parallel attention layers
            d_ff (int): FFN hidden dim (4 * d_h)
            rate (float): dropout rate
            layer_idx (int): layer index
        """
        super().__init__()

        self.attn = MultiHeadAttention(d_h=d_h,
                                       head=head,
                                       rate=rate,
                                       layer_idx=layer_idx)

        self.ffn = FeedForward(d_h=d_h,
                               d_ff=d_ff,
                               rate=rate)

        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, 
             x: tf.Tensor, 
             mask: tf.Tensor,
             training: bool):
        """forward propagation

        Args:
            x (tf.Tensor(bz, dec_len, d_h)): input
            mask (tf.Tensor(bz, 1, dec_len, dec_len)): mask
            training (bool): train or not

        Returns:
            output (tf.Tensor(bz, dec_len, d_h)): output
        """

        # Masked Multi-Head Attention
        attn = self.norm_1(x)
        attn = self.attn(attn, attn, attn, mask, training)
        attn = x + attn

        # Feed Forward
        ffn = self.norm_2(attn)
        ffn = self.ffn(ffn, training)
        output = ffn + attn

        return output