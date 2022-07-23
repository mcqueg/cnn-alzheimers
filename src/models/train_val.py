import os
import random

from sympy import evaluate
from yaml import load
from src.models.model_utils import plot_history
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model

def shuffle_dirs(parent_dir):
    # shuffle the subdirectories of the training parent dir before training starts
    random.seed(30)
    for dir in os.listdir(parent_dir):
        random.shuffle(os.listdir(os.path.join(parent_dir, dir)))

def train_val_model(model,
                    model_name,
                    train_dir,
                    test_dir,
                    logs_dir,
                    save_dir,
                    val_size,
                    epochs,
                    batch_size,
                    plot_history=False,
                    evaluate=False):
    '''
    Purpose: Train the compiled model. Model trains for specified epochs or until it early
                stopping is triggered via a val_loss call back. 
    parameters:
        model - compiled motel to be trained.  
        train_dir- string - root dir of the training images to use. organized by class in foflders.
        val_size - float - size of the validation split from the training data
                    epochs num of epochs to train for
        batch_size - int - batch size of images
    Returns:
        history - model training history
    '''
    now = datetime.now()
    start_time = now.strftime("_%d-%m-%Y_%H:%M:%S")
    
    m = '*'
    print(f"\n{m*70}\n\t\tSHUFFLING IMAGES\n{m*70}\n")
    # shuffle images
    shuffle_dirs(train_dir)
    # -- BUILD IMAGE GENERATORS -- 
    train_datagenerator = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=45,
        brightness_range=(0,0.2),
        zoom_range=0.2,
        horizontal_flip = True,
        fill_mode='nearest',
        validation_split = val_size)
    
    test_datagenerator = ImageDataGenerator(
        rescale=1./255
    )

    train_gen = train_datagenerator.flow_from_directory(train_dir,
                                                  batch_size = batch_size,
                                                  class_mode = 'categorical',
                                                  subset='training',
                                                  shuffle=True,
                                                  seed = 42)
    
    val_gen = train_datagenerator.flow_from_directory(train_dir,
                                                  batch_size = batch_size,
                                                  class_mode = 'categorical',
                                                  subset='validation',
                                                  shuffle=True,
                                                  seed = 42)

    test_gen = test_datagenerator.flow_from_directory(test_dir,
                                                     batch_size = batch_size,
                                                     class_mode = 'categorical')

    # -- CREATE CALLBACKS --
    # create new unique log dir for current run log for Tensorboard
    name = f'{model_name}_{start_time}'
    log = os.path.join(logs_dir, name)
    # generate callbaks
    callbacks = [
        # monitor the validation loss exiting training if it doesnt improve
        #   after 'patience' num of epochs.
        EarlyStopping(monitor='val_loss', mode='min',patience=10, verbose=2),

        # monitor training progress using Tensorboard
        TensorBoard(log_dir = log),

        # checkpoint model 
        ModelCheckpoint(
            filepath=os.path.join(save_dir,name),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='min',
            save_freq='epoch')
        ]

    # -- FIT MODEL -- 
    history = model.fit(train_gen,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_gen,
                        callbacks = [callbacks])

    if plot_history:
        print("\nPlotting training history ...\n")
        plot_history(history)

    # -- EVALUATE MODEL -- 
    if evaluate:
        m = '*'
        print(f"\n{m*70}\n\t\tEVALUATING BEST MODEL\n{m*70}\n")
        # load best model (latest save)
        model_best = load_model(os.path.join(save_dir,name))

        loss, acc = model_best.evaluate(test_gen,
                                   batch_size=batch_size,
                                   verbose=2)

        print("MODEL TEST ACCURACY: {:5.2f}%".format(100 * acc))
        print("MODEL TEST LOSS: {:5.2}%".format(loss))
        

    return history