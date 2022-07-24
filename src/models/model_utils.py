
from tensorflow.keras import layers
from tensorflow.keras import Model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

# # local imports
# from src.data.process import process_img

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
def load_model(json_path,
               weights_path,
               last_frozen_layer,
               class_num,
               print_summary=False):

    # load archhitecture

    # load weights

    # get layer index for the last frozen layer

    # iterate through the last frozen layer freezing weights

    # clone model randomly initializing the weights

    # set output of frozen model to be the last forzen layer

    # set input of cloned model to be the last_frozen layer index +1

    # build model from both

    # set ouput of that model to be len(model.layers) - 1

    # feed output to a new classificaiton layer based on class num

    if print_summary:
        model.summary()

    return model

#---------------------------------------------------------------------------------------------------

def plot_history(history):
    '''Plots the training and validation loss and accuracy from a history object'''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()