import tensorflow as tf
from tensorflow.keras import layers, Model

from model.embed import positional_encoding
from model.layer import DecoderLayer

from utils.metrics import custom_accuracy, custom_ppl
from utils.losses import custom_cross_entrpy


class DialoGPT(Model):

    def __init__(self, 
                 vocab_size: int,
                 max_len: int,
                 d_h: int, 
                 head: int, 
                 d_ff: int, 
                 rate: float,
                 n_layer: int):
        """DialoGPT

        Args:
            vocab_size (int): vocabulary size
            max_len (int): max length
            d_h (int): attn hidden dim(=embedding dim)
            head (int): parallel attention layers
            d_ff (int): FFN hidden dim
            rate (float): dropout rate
            n_layer (int): number of layer
        """
        super().__init__()

        self.embed = layers.Embedding(vocab_size, d_h)
        self.pos_enc = positional_encoding(max_len=max_len,
                                           d_emb=d_h)
        self.dropout = layers.Dropout(rate)

        self.dec_layers = [DecoderLayer(d_h=d_h,
                                        head=head,
                                        d_ff=d_ff,
                                        rate=rate,
                                        layer_idx=i) for i in range(n_layer)]

        self.proj = layers.Dense(vocab_size, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
                                                    
    def call(self, x: tf.Tensor, training: bool):
        """forward propagation

        Args:
            x (tf.Tensor(bz, max_len)): input
            training (bool): train or not

        Returns:
            output (tf.Tensor(bz, max_len, vocab_size)): output
        """

        #diff between train and inference
        input_len = x.shape[1]
        
        mask = self.get_mask(x)

        x = self.embed(x) + self.pos_enc[:, :input_len]

        x = self.dropout(x)

        for layer in self.dec_layers:
            x = layer(x, mask, training)
        
        output = self.norm(self.proj(x))

        return output

    def get_mask(self, x: tf.Tensor):
        """get uttr look ahead mask 

        Args:
            x (tf.Tensor(bz, max_len)): input

        Returns:
            mask (tf.Tensor(bz, 1, uttr_len, uttr_len)): mask
        """

        #diff between train and inference
        input_len = x.shape[1]

        # pad mask: bz x uttr_len -> bz x 1 x 1 x uttr_len
        pad_mask = tf.cast(tf.math.equal(x, 0), dtype=tf.float32)[:, tf.newaxis, tf.newaxis, :]
        # attn mask: uttr_len x uttr_len
        attn_mask = 1 - tf.linalg.band_part(tf.ones((input_len, input_len)), -1, 0)
        
        mask_uttrs = tf.maximum(pad_mask, attn_mask)

        return mask_uttrs


class DialogTrain(Model):

    def __init__(self, 
                 vocab_size: int,
                 max_len: int,
                 d_h: int, 
                 head: int, 
                 d_ff: int, 
                 rate: float,
                 n_layer: int):
        """Train DialoGPT

        Args:
            vocab_size (int): vocabulary size
            max_len (int): max length
            d_h (int): attn hidden dim(=embedding dim)
            head (int): parallel attention layers
            d_ff (int): FFN hidden dim
            rate (float): dropout rate
            n_layer (int): number of layer
        """
        super().__init__()

        self.dialog = DialoGPT(vocab_size=vocab_size,
                               max_len=max_len,
                               d_h=d_h,
                               head=head,
                               d_ff=d_ff,
                               rate=rate,
                               n_layer=n_layer)

    def call(self, inputs, training: bool):
        """forward propagation
        
        Args:
            inputs: (
                input (tf.Tensor(bz, max_len)): input
                label (tf.Tensor(bz, max_len)): label
            )
           training (bool): train or not

           output (tf.Tensor(bz, max_len, vocab_size)): output
        """

        # bz x len -> bz x len x vocab_size
        predict = self.dialog(inputs['input'], training)

        loss = custom_cross_entrpy(inputs['label'], predict)

        self.add_loss(loss)

        # Calculate metrics
        prediction = tf.cast(tf.argmax(predict, axis=-1), dtype=tf.float32)
        acc = custom_accuracy(inputs['label'], prediction)
        ppl = custom_ppl(inputs['label'], predict)
        self.add_metric(ppl, name='ppl')
        self.add_metric(acc, name='acc')

        return predict