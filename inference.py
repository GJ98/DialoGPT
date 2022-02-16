import json
import tensorflow as tf
import sentencepiece as spm
from hydra import initialize, compose
from tensorflow import keras
import numpy as np
from eunjeon import Mecab

from utils.data import token_to_sentence
from model.chatbot import *


def inference(config_name: str, 
              config_path: str, 
              save_path: str, 
              vocab_path: str, 
              dialog_path: str):
    """inference
    
    Args:
        config_name (str): config file name
        config_path (str): config path
        save_path (str): dialoGPT weights path
        vocab_path (str): vocabulary path
        data_path (str): sample data path
    """
    initialize(config_path)
    cfg = compose(config_name)

    # MODEL
    model = keras.models.load_model(save_path)

    type = input('MODEL TYPE\n' +
    '  1. GreedyChatbot\n' +
    '  2. BeamChatbot\n' +
    '  3. CheatAllChatbot\n' +
    '  4. CheatFirstChatbot\n:')

    if type == '1':
        chatbot = GreedyChatbot(model, cfg.processing.max_len)
    elif type == '2':
        print("Not implement")
        return
    elif type == '3':
        print("Not implement")
        return
    elif type == '4':
        print("Not implement")
        return
    else:
        return 

    # TOKENIZER
    mecab = Mecab()
    sp = spm.SentencePieceProcessor()
    sp.Load(vocab_path)

    # DATA
    dialog_f = open(dialog_path, "r", encoding="utf-8")
    dialogs = json.load(dialog_f)

    # INFERENCE
    while True:
        i = input('num (exit == -1): ')
        i = int(i)
        if i == -1: exit()

        if type == "1":
            source = []
            for uttr in dialogs[i]:
                source.append(uttr)
                idx = sp.EncodeAsIds(" ".join(source))
                idx = tf.convert_to_tensor(np.array(idx), dtype=tf.float32)
                resp = tf.cast(chatbot(idx), dtype=tf.int32)

                print("Input")
                for i, u in enumerate(source):
                    print(f"uttr{i}: {u}")

                print("Output")
                predict = sp.IdToPiece(resp.numpy().tolist())
                predict = token_to_sentence(predict)
                print(predict)

if __name__=="__main__":
    version = "small"
    config_name = f"{version}.yaml"
    config_path = "./configs"
    save_path = f"./save/{version}0"
    vocab_path = "./data/emotion1/spm.model"
    data_path = "./data/emotion1/dialog_val.json"
 
    inference(config_name, config_path, save_path, vocab_path, data_path)