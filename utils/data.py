import os, json, re
import csv
import numpy as np
import sentencepiece as spm
import tensorflow as tf

from typing import Dict, List
from random import shuffle, randrange, random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from eunjeon import Mecab

def chatbot_to_dialog(chatbot_path: str, dialog_path: str):
    """build chatbot data into dialog data
        
    Args:
        chatbot_path (str): chatbot data file(.json) path
        dialog_path (str): dialog data file(.json) path 
    """

    chatbot_f = open(chatbot_path, 'r', encoding='utf-8')
    dialog_f = open(dialog_path, "w", encoding="utf-8")

    chatbot = csv.reader(chatbot_f)
    lines = list(chatbot)
    row_size= len(lines)

    dialog_f.write("[")
    for i, line in tqdm(enumerate(lines), desc="building dialogs..."):
        if i == 0: continue
        dialog = [line[0], line[1]]

        dialog_f.write(json.dumps(dialog, ensure_ascii=False))

        if i != row_size - 1: dialog_f.write(",\n")
    dialog_f.write("]")

    chatbot_f.close()
    dialog_f.close()

def emotion_to_dialog(emotion_path: str, dialog_path: str):
    """build emotion data into dialog data
        
    Args:
        emotion_path (str): emotion data file(.json) path
        dialog_path (str): dialog data file(.json) path 
    """

    emotion_f = open(emotion_path, 'r', encoding='utf-8')
    dialog_f = open(dialog_path, 'w', encoding='utf-8')

    lines = json.load(emotion_f)

    dialog_f.write("[")
    for i, line in tqdm(enumerate(lines), desc="building dialogs..."):
        dialog = [
            line["talk"]["content"]["HS01"],
            line["talk"]["content"]["SS01"],
            line["talk"]["content"]["HS02"],
            line["talk"]["content"]["SS02"],
            line["talk"]["content"]["HS03"],
            line["talk"]["content"]["SS03"]
        ]
        while "" in dialog: dialog.remove("")

        dialog_f.write(json.dumps(dialog, ensure_ascii=False))

        if i != len(lines) - 1: dialog_f.write(",\n")
    dialog_f.write("]")

    emotion_f.close()
    dialog_f.close()


def dialog_to_inst(vocab_path: str,
                   dialog_path: str,
                   inst_path: Dict,
                   max_len: int):
    """ convert dialog to inst

    Args:
        vocab_path (str): vocabulary file(.json) path 
        dialog_path (str): dialog file(.json) path 
        inst_path ({'idx': str, 'token': str}): instance file(.json) path
        max_len (int): max length
    """
        
    # 0. Load vocabulary
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(vocab_path)

    # 1. dialog -> idx
    dialog_f = open(dialog_path, "r", encoding="utf-8")
    dialogs = json.load(dialog_f)

    idxs = []
    for dialog in tqdm(dialogs, desc=f"Tokenizing {dialog_path}..."):
        idx = {
            'source': tokenizer.EncodeAsIds(dialog[0]),
            'target': tokenizer.EncodeAsIds(" ".join(dialog[1:]))
        }
        idxs.append(idx)

    dialog_f.close()

    # 2. build inst.json(idx -> instance)
    idx_f = open(inst_path['idx'], "w", encoding="utf-8")
    token_f = open(inst_path['token'], "w", encoding="utf-8")

    idx_f.write("["),  token_f.write("[")
    for idx in tqdm(idxs, desc="Instancing..."):
        input = idx['source'] + idx['target'] + [3]
        label = [0] * (len(idx['source']) - 1) + idx['target'] + [3, 0]

        # padding
        if len(input) < max_len:
            input = input + [0 for _ in range(max_len - len(input))]
            label = label + [0 for _ in range(max_len - len(label))]

        token = {
            "input": tokenizer.IdToPiece(input),
            "label": tokenizer.IdToPiece(label),
        }

        _idx = {
            "input": input,
            "label": label,
        }

        idx_f.write(json.dumps(_idx, ensure_ascii=False))
        token_f.write(json.dumps(token, ensure_ascii=False))

        if idx != idxs[len(idxs) - 1]:    
            idx_f.write(",\n")
            token_f.write(",\n")
    idx_f.write("]")
    token_f.write("]")

    idx_f.close()
    token_f.close()


def build_vocab(cntxt_path: str, vocab_path: str, vocab_size: int):
    """build vocab
        
    Args:
        cntxt_path (str): context data file(.json) path 
        vocab_path (str): vocabulary file(.json) path 
        vocab_size (int): vocabulary size
    """

    mecab = Mecab()
    cntxt_f1 = open(cntxt_path['train'], "r", encoding="utf-8")
    cntxt_f2 = open(cntxt_path['val'], "r", encoding="utf-8")
    morphs_f = open("./morphs.txt", "w", encoding="utf-8")

    train_lines = json.load(cntxt_f1)
    val_lines = json.load(cntxt_f2)

    # 1. morphs
    cntxts = []
    # 1-1. train data
    for i, line in tqdm(enumerate(train_lines), desc=f"building training morpheme..."):
        cntxts.extend(line)
        if i % 1000 == 0 or i + 1 == len(train_lines):
            morphs = mecab.morphs(" ".join(cntxts))
            morphs_f.write(" ".join(morphs))
            cntxts = []
    
    # 2-1. val data
    for i, line in tqdm(enumerate(val_lines), desc=f"building val morpheme..."):
        cntxts.extend(line)
        if i % 1000 == 0 or i + 1 == len(val_lines):
            morphs = mecab.morphs(" ".join(cntxts))
            morphs_f.write(" ".join(morphs))
            cntxts = []

    # 2. sentencepiece
    spm.SentencePieceTrainer.train(
        f"--input=./morphs.txt --model_prefix={vocab_path} --vocab_size={vocab_size}" + 
        " --model_type=bpe" +
        " --max_sentence_length=99999999" + # max length
        " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
        " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
        " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
        " --eos_id=3 --eos_piece=[EOS]"   # end of sequence (3)
    ) 

    os.remove('./morphs.txt')

def get_chatbot_dataset(inst_file: str, batch_size: int):
    """get chatbot numpy dataset
    
    Args:
        inst_file (str): instance file(.json) path
        (default="./data/chatbot_data/inst.json")
        batch_size (int): batch size
    """

    if not os.path.isfile(inst_file):
        raise Exception (f"{inst_file} doesn't exist")

    inputs = []
    labels = []

    inst_f = open(inst_file, "r", encoding="utf-8")
    insts = json.load(inst_f)
    for inst in tqdm(insts, desc="Loading instances..."):
        inputs.append(np.array(inst["input"], dtype=np.float32))
        labels.append(np.array(inst["label"], dtype=np.float32))
    
    inst_f.close()

    x_train, x_val, y_train, y_val = \
        train_test_split(np.array(inputs), np.array(labels), test_size=0.2, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                'input': x_train,
                'label': y_train,
            }
        )
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                'input': x_val,
                'label': y_val,
            }
        )
    )

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    return train_dataset, val_dataset


def get_emotion_dataset(inst_file: str, batch_size: int):
    """get emotion numpy dataset
    
    Args:
        inst_file (str): instance file(.json) path
        (default="./data/emtion/inst.json")
        batch_size (int): batch size
    """

    if not os.path.isfile(inst_file):
        raise Exception (f"{inst_file} doesn't exist")

    inputs = []
    labels = []

    inst_f = open(inst_file, "r", encoding="utf-8")
    insts = json.load(inst_f)
    for inst in tqdm(insts, desc="Loading instances..."):
        inputs.append(np.array(inst["input"], dtype=np.float32))
        labels.append(np.array(inst["label"], dtype=np.float32))
    
    inst_f.close()

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                'input': inputs,
                'label': labels,
            }
        )
    )
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    return dataset


def token_to_sentence(token: List[str]):
    """convert token to sentence
    
    Args:
        token (List[str]): token list
    """
    token = "".join(token)
    token = token.replace("▁", " ")
    token = token.replace("[CLS]", "")
    token = token.replace("[PAD]", "")
    token = re.sub("\[EOS\].*", "", token)
    sentence = re.sub("\[SEP\].*", "", token)

    return sentence


if __name__=='__main__':
    """1. primitive to dialog
    primitive_path = "../data/emotion0/primitive_train.json"
    dialog_path = "../data/emotion1/dialog_train.json"
    emotion_to_dialog(primitive_path, dialog_path)
    """

    """2. build vocab
    dialog_path = {
        'train': "../data/emotion1/dialog_train.json",
        'val': "../data/emotion1/dialog_val.json"
    }
    vocab_path = "../data/emotion1/spm"
    vocab_size = 32000
    build_vocab(dialog_path, vocab_path, vocab_size)
    """

    """3. preprocess data
    vocab_path = "../data/emotion1/spm.model"
    cntxt_path = "../data/emotion1/dialog_val.json"
    inst_path = {
        "idx": "../data/emotion1/idx_val.json",
        "token": "../data/emotion1/token_val.json"
    }
    max_len = 250
    # emotion max length: 250

    dialog_to_inst(vocab_path, cntxt_path, inst_path, max_len)
    """

    """sampling preprocessor data"""
    inst_path = "../data/emotion1/token_val.json"
    inst_f = open(inst_path, "r", encoding="utf-8")
    insts = json.load(inst_f)

    a = [i for i in range(5000)]
    shuffle(a)
    print(a[0])
    sample = insts[a[0]]
    print(f'\ninput: {sample["input"]}')
    print(f'\nlabel: {sample["label"]}')

    print(f'\ninput len: {len(sample["input"])}')
    print(f'\nlabel len: {len(sample["label"])}')
    inst_f.close()