from distutils.log import Log
import os
from datetime import datetime
import time
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

# local imports
from src.data.process import process_img

def add_Dense_layers(model, last_output, dense_nodes, class_num, dropout):
    '''
    Purpose:
        To append a trainable dense network to the specified output layer of 
        the model with frozen weights. The network has two layers one dense layer
        with ReLU activation and a classification layer w/ softmax activation.
        These layers allow for the pretrained weights to be applied on a new dataset. 
    Parameters:
        model - the pretrained model to append the dense network to
        last_output- the output from the last layer of the pretrained model.
                      this will be fed into the dense network
        dense_nodes - list- determines the number of nodes to include in the dense layers
        class_num-int- determines the number of output nodes are needed in the classification layer.
                    Note: this should be set to the number of predicted classes.
        droput-float- specified dropout to apply to the dense layer. 
    '''
    print('Initializing Dense layers...')
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)

    for num in dense_nodes:
        # add a fully connected layer
        x = layers.Dense(num, activation='relu')(x)
        # add the dropout rate of 0.2
        x = layers.Dropout(dropout)(x)
    # add a final layer with softmax for classification
    x = layers.Dense(class_num, activation='softmax')(x)

    # append the created dense network to the pre_trained_model
    dense_model = Model(model.input, x)

    return dense_model

    #--------------------------------------------------------------------------------------------

def train_val_model(model,
                    model_name,
                    train_dir,
                    logs_dir,
                    save_dir,
                    val_size,
                    epochs,
                    batch_size):
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
    # -- BUILD IMAGE GENERATORS -- 
    datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=20,
        horizontal_flip = True,
        vertical_flip = True,
        validation_split = val_size)

    train_generator = datagen.flow_from_directory(train_dir,
                                                  batch_size = batch_size,
                                                  class_mode = 'categorical',
                                                  subset='training',
                                                  shuffle=True,
                                                  seed = 42)
    
    val_generator = datagen.flow_from_directory(train_dir,
                                                  batch_size = batch_size,
                                                  class_mode = 'categorical',
                                                  subset='validation',
                                                  shuffle=True,
                                                  seed = 42)

    # -- CREATE CALLBACKS --
    # create new unique log dir for current run log for Tensorboard
    name = f'{model_name}_{start_time}'
    log = os.path.join(logs_dir, name)
    # generate callbaks
    callbacks = [
        # monitor the validation loss exiting training if it doesnt improve
        #   after 'patience' num of epochs.
        EarlyStopping(monitor='val_loss', mode='min',patience=10, verbose=1),

        # monitor training progress using Tensorboard
        TensorBoard(log_dir = log),

        # checkpoint model 
        ModelCheckpoint(
            filepath=os.path.join(save_dir,name),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_freq='epoch')
        ]

    # -- FIT MODEL -- 
    history = model.fit(train_generator,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_generator,
                        callbacks = [callbacks])

    #plot_history(history)

    return history

#---------------------------------------------------------------------------------------------------

def plot_history(history):

    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs=range(len(acc)) # Get number of epochs

    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.show()
    print("")

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")
    plt.show()