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
    name = f'{model_name}_{start_time}'
    log = os.path.join(logs_dir, name)
    weights_path=os.mkdirs(os.path.join(os.path.join(save_dir,name),'weights'))
    print(f"\nsaving weights at: {weights_path}")
    config_path=os.mkdirs(os.path.join(os.path.join(save_dir,name), 'config'))
    print(f"\nsaving model architecture at: {config_path}")
    # generate callbaks
    callbacks = [
        # monitor the validation loss exiting training if it doesnt improve
        #   after 'patience' num of epochs.
        EarlyStopping(monitor='val_loss', mode='min',patience=10, verbose=2),

        # monitor training progress using Tensorboard
        TensorBoard(log_dir = log),

        # checkpoint model 
        ModelCheckpoint(
            filepath=weights_path,
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

    # -- save model architecture
    model_config = model.to_json()
    with open(os.path.join(config_path, f'{name}.json'), 'w') as outfile:
        json.dump(model_config, outfile)

    if plot_history:
        print("\nPlotting training history ...\n")
        plot_history(history)

    # -- EVALUATE MODEL -- 
    if evaluate:
        m = '*'
        print(f"\n{m*70}\n\t\tEVALUATING BEST MODEL\n{m*70}\n")
        # load best model (latest save)
        weights_path = os.path.join(save_dir,name)
        model_eval = model_from_json(os.path.join(config_path, f'{name}.json'))
        model_eval.load_weights(weights_path)
        # if model_idx == 0:
        #     model_best = load_inception_v3(weights_path, input_shape, last_layer, dense_nodes, class_num, dropout)
        model_eval.compile(optimizer = Adam(learning_rate=lr),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

        loss, acc = model_eval.evaluate(test_gen,
                                   batch_size=batch_size,
                                   verbose=2)

        print("MODEL TEST ACCURACY: {:5.2f}%".format(100 * acc))
        print("MODEL TEST LOSS: {:5.2}%".format(loss))
        

    return history