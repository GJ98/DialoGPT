import wandb
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import optimizers, callbacks
from wandb.keras import WandbCallback
from hydra import initialize, compose

from model.model import DialogTrain
from utils.data import get_emotion_dataset, get_chatbot_dataset


def pretrain(config_path: str, config_name: str, data_path: str, save_path: str):
    """pretrain
    
    Args:
        config_name (str): config file name
        config_path (str): config path
        data_path (str): train and val data path
        save_path (str): dialoGPT save path
    """

    # Wandb & Hydra
    initialize(config_path)
    cfg = compose(config_name)
    #wandb.init(project="DialoGPT", entity="gj98", config=cfg)

    # DATASET
    train_dataset = get_emotion_dataset(data_path['train'], cfg.processing.batch_size)
    val_dataset = get_emotion_dataset(data_path['val'], cfg.processing.batch_size)

    print(f'train dataset: {train_dataset}\n')
    print(f'val dataset: {val_dataset}\n')
    
    # MODEL
    model = DialogTrain(
        cfg.model.vocab_size,
        cfg.processing.max_len,
        cfg.model.d_h,
        cfg.model.head,
        cfg.model.d_ff,
        cfg.processing.rate,
        cfg.model.n_layer,
    )

    # COMPILE
    model.compile(
        optimizer=optimizers.Adam(
            cfg.training.learning_rate,
            beta_1=cfg.training.beta1,
            beta_2=cfg.training.beta2,
            epsilon=cfg.training.eps),
    )

    # TRAIN
    model.fit(
        train_dataset,
        epochs=cfg.training.epoch,
        validation_data=val_dataset,
        callbacks=[
            #WandbCallback(), 
            callbacks.ReduceLROnPlateau(patience=1),
            callbacks.EarlyStopping(patience=3)
        ]
    )

    # SAVE
    model.dialog.save(save_path)



def pretrain_tpu(config_path: str, config_name: str, data_path: str, save_path: str):
    """pretrain
    
    Args:
        config_name (str): config file name
        config_path (str): config path
        data_path (str): data path
        save_path (str): DialoGPT save path
    """

    # TPU
    print("Tensorflow version " + tf.__version__)

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

    # Wandb & Hydra
    initialize(config_path)
    cfg = compose(config_name)
    wandb.init(project="DialoGPT", entity="gj98", config=cfg)

    # DATASET
    train_dataset = get_emotion_dataset(data_path['train'], cfg.processing.batch_size)
    val_dataset = get_emotion_dataset(data_path['val'], cfg.processing.batch_size)

    print(f'train dataset: {train_dataset}\n')
    print(f'val dataset: {val_dataset}\n')
    
    with tpu_strategy.scope():
        # MODEL
        model = DialogTrain(
            cfg.model.vocab_size,
            cfg.processing.max_len,
            cfg.model.d_h,
            cfg.model.head,
            cfg.model.d_ff,
            cfg.processing.rate,
            cfg.model.n_layer,
        )

        # COMPILE
        model.compile(
            optimizer=optimizers.Adam(
                cfg.training.learning_rate,
                beta_1=cfg.training.beta1,
                beta_2=cfg.training.beta2,
                epsilon=cfg.training.eps),
        )

    # TRAIN
    model.fit(
        train_dataset,
        epochs=cfg.training.epoch,
        validation_data=val_dataset,
        callbacks=[
            WandbCallback(), 
            callbacks.ReduceLROnPlateau(patience=1),
            callbacks.EarlyStopping(patience=3),
            callbacks.ModelCheckpoint(save_path, monitor='val_ppl', options=localhost_save_option, save_best_only=True)
        ]
    )


if __name__=="__main__":
    version = "small"
    config_name = f"{version}.yaml"
    config_path = "./configs"
    data_path = {
        'train': "./data/emotion1/idx_train.json",
        'val': "./data/emotion1/idx_val.json"
    }
    save_path = f"./save/{version}"

    type = input("type of device: ")
    if type == 'gpu' or type == 'GPU' or type == 'cpu' or type == 'CPU':
        pretrain(config_path, config_name, data_path, save_path)
    elif type == 'tpu' or type == 'TPU':
        pretrain_tpu(config_path, config_name, data_path, save_path)
    else:
        print(f"{type} device doesn't exists")