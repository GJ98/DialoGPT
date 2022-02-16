import tensorflow as tf
from tensorflow.keras import layers


class MultiHeadAttention(layers.Layer):

    def __init__(self, d_h: int, head: int, rate: float, layer_idx: int):
        """multi head attention

        Args:
            d_h (int): hidden dim
            head (int): parallel attention layers
            rate (float): dropout rate
            layer_idx (int): layer index
        """
        super().__init__()

        assert d_h % head == 0

        self.d_h, self.head = d_h, head
        self.d_attn = d_h // head
        self.layer_idx = layer_idx

        self.w_v = layers.Dense(d_h, use_bias=False)
        self.w_k = layers.Dense(d_h, use_bias=False)
        self.w_q = layers.Dense(d_h, use_bias=False)
        
        self.w_o = layers.Dense(d_h)

        self.dropout_1 = layers.Dropout(rate=rate)
        self.dropout_2 = layers.Dropout(rate=rate)

    def split_head(self, x: tf.Tensor, seq_len: int):
        """split v, k, q head

        Args:
            x (tf.Tensor(bz, len, d_h)): key, val, query
            seq_len (int): sequence length

        Returns:
            splited x: tf.Tensor(bz, head, len, d_attn)
        """

        x = tf.reshape(x, (-1, seq_len, self.head, self.d_attn))
        return tf.transpose(x, perm=[0, 2, 1 ,3])

    def attention(self, 
                  q: tf.Tensor, 
                  k: tf.Tensor, 
                  v: tf.Tensor, 
                  mask: tf.Tensor,
                  training: bool):
        """scaled dot product attention
        
        Args:
            q (tf.Tensor(bz, head, len_q, d_k)): query
            k (tf.Tensor(bz, head, len_k, d_k)): key
            v (tf.Tensor(bz, head, len_k, d_v)): value
            mask (tf.Tensor(bz, 1, :, len_k)): mask 
            training (bool): train or not

        Returns:
            output (tf.Tensor(bz, head, len_q, d_v)): output
        """

        weight = tf.matmul(q, k, transpose_b=True) / \
            tf.math.sqrt(tf.cast(self.d_attn, dtype=tf.float32))

        if mask is not None:
            weight += (mask * -1e9)

        weight = weight / tf.cast(self.layer_idx + 1, dtype=tf.float32)

        scale_weight = tf.nn.softmax(weight)

        scale_weight = self.dropout_1(scale_weight, training=training)

        output = tf.matmul(scale_weight, v)

        return output

    def call(self, 
             v: tf.Tensor, 
             k: tf.Tensor,
             q: tf.Tensor,
             mask: tf.Tensor,
             training: bool): 
        """forward propagation

        Args:
            v (tf.Tensor(bz, len_k, d_h)): value
            k (tf.Tensor(bz, len_k, d_h)): key
            q (tf.Tensor(bz, len_q, d_h)): query
            mask (tf.Tensor(bz, 1, :, len_k)): mask 
            training (bool): train or not
        
        Returns:
            output (tf.Tensor(bz, len_q, d_h)): output
        """

        #diff between train and inference
        len_k, len_q = k.shape[1], q.shape[1]

        v = self.split_head(self.w_v(v), len_k)
        k = self.split_head(self.w_k(k), len_k)
        q = self.split_head(self.w_q(q), len_q)
        
        attn = self.attention(q, k, v, mask, training)

        attn = tf.reshape(tf.transpose(attn, perm=[0, 2, 1, 3]), (-1, len_q, self.d_h))

        output = self.w_o(attn)

        output = self.dropout_2(output, training=training)

        return output


class FeedForward(layers.Layer):
    
    def __init__(self, d_h: int, d_ff: int, rate: float):
        """position wise feed forward
        
        Args:
            d_h (int): attn hidden dim
            d_ff (int): FFN hidden dim
            rate (float): dropout rate
        """
        super().__init__()
    
        self.w_1 = layers.Dense(d_ff, activation='gelu')
        self.w_2 = layers.Dense(d_h)
        
        self.dropout = layers.Dropout(rate=rate)

    def call(self, x: tf.Tensor, training: bool):
        """forward propagation
        
        Args:
            x (tf.Tensor(bz, len, d_h)): input
            training (bool): train or not
        
        Returns:
            output (tf.Tensor(bz, len, d_h)): output
        """

        x = self.w_1(x)

        output = self.w_2(x)

        output = self.dropout(output, training=training) 

        return output