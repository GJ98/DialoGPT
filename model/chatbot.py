import tensorflow as tf
from tensorflow.keras import Model


class GreedyChatbot(tf.Module):

    def __init__(self, 
                 dialog: Model,
                 max_len: int):
        """DialoGPT Greedy search Chatbot
        
        Args:
            dialog (tf.keras.Model): dialoGPT
            max_len (int): max length
        """
        super().__init__()

        self.dialog = dialog
        self.max_len = max_len

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def __call__(self, input: tf.Tensor):
        """forward propagation
        
        Args:
            input (tf.Tensor(len)): input

        Returns:
            output (tf.Tensor): response
        """

        # 0. initialize resp
        x = tf.TensorArray(dtype=tf.float32, size=self.max_len)
        x = x.scatter(tf.range(tf.size(input)), input)

        # 1. predict response
        for i in tf.range(tf.size(input), self.max_len):
            predict = self.dialog(x.stack()[tf.newaxis, :])
            last_word = tf.cast(tf.argmax(predict[0, i - 1], axis=-1), dtype=tf.float32)
            x = x.write(i, last_word)
            if last_word == tf.constant(31051, dtype=tf.float32) or \
                last_word == tf.constant(31071, dtype=tf.float32) or \
                last_word == tf.constant(31249, dtype=tf.float32):
                break
        
        return tf.cast(x.gather(tf.range(tf.size(input), self.max_len)), dtype=tf.int32)