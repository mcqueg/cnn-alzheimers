import os
import json
import random
from src.models.model_utils import plot_history
from src.models.inception_v3 import build_inception_v3
# from src.models.vgg19 import build_vg19
# from src.models.resnet50 import build_net50
# from src.models.xception import build_xception
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import load_weights
from tensorflow.keras.models import model_from_json
from keras.initializers import glorot_uniform



def shuffle_dirs(parent_dir):
    # shuffle the subdirectories of the training parent dir before training starts
    random.seed(30)
    for dir in os.listdir(parent_dir):
        random.shuffle(os.listdir(os.path.join(parent_dir, dir)))

def build_train_val_model(input_shape,
                    class_num,
                    last_layer,
                    dense_nodes,
                    lr,
                    dropout,
                    model_name,
                    train_dir,
                    test_dir,
                    logs_dir,
                    save_dir,
                    val_size,
                    epochs,
                    batch_size,
                    print_summary=False,
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
    name = f'{model_name}_{start_time}'
    
    # -- BUILDING MODEL -- 
    m = '*'
    print(f"\n{m*60}\n\t\tBUILDING MODEL\n{m*60}\n")
    model = build_inception_v3(input_shape,
                       class_num,
                       last_layer,
                       dense_nodes,
                       dropout, 
                       print_summary=print_summary
                       )

    os.makedirs(os.path.join(os.path.join(save_dir,name),'ckpt'), exist_ok=True)
    weights_path=os.path.join(os.path.join(save_dir,name), 'ckpt/')
    print(f"\nsaving weights at: \n\t{weights_path}")

    os.makedirs(os.path.join(os.path.join(save_dir,name), 'config'), exist_ok=True)
    config_path=os.path.join(os.path.join(save_dir,name), 'config')
    print(f"\nsaving model architecture at: \n\t{config_path}\n")
    
    # -- save model architecture
    model_config = model.to_json()
    with open(os.path.join(config_path, f'{name}.json'), 'w') as json_file:
        json_file.write(model_config)


    print(f"\n{m*60}\n\t\tCOMPILING MODEL\n{m*60}\n")
    # compile model
    model.compile(optimizer = Adam(learning_rate=lr),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    # BEGIN TRAINING
    print(f"\n{m*60}\n\t\tSHUFFLING IMAGES\n{m*60}\n")
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

    val_datagenerator = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_size
    )
    
    test_datagenerator = ImageDataGenerator(
        rescale=1./255
    )

    train_gen = train_datagenerator.flow_from_directory(train_dir,
                                                  batch_size = batch_size,
                                                  class_mode = 'categorical',
                                                  subset='training',
                                                  shuffle=True,
                                                  seed = 42)
    
    val_gen = val_datagenerator.flow_from_directory(train_dir,
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
            filepath=os.path.join(weights_path, f'{name}_weights.h5'),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
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
        print(f"\n{m*70}\n\t\tEVALUATING MODEL\n{m*70}\n")
        # load best model (latest save)
        config = os.path.join(os.path.join(save_dir,name), 'config')
        config_path = f'{config}/{name}.json'
        with open (config_path, 'r') as json_file:
            json_Model = json_file.read()
        model_eval = model_from_json(json_Model)

        # load the weights
        weights_path = os.path.join(weights_path, f'{name}_weights.h5')
        model_eval.load_weights(weights_path)
        # if model_idx == 0:
        #     model_best = load_inception_v3(weights_path, input_shape, last_layer, dense_nodes, class_num, dropout)
        model_eval.compile(optimizer = Adam(learning_rate=lr),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

        loss, acc = model_eval.evaluate(test_gen,
                                   batch_size=batch_size,
                                   verbose=2)

        print("\nMODEL TEST ACCURACY: {:5.2f}%".format(100 * acc))
        print("\nMODEL TEST LOSS: {:5.2}%".format(loss))
        

    return history