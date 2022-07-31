
from tensorflow.keras import layers
from tensorflow.keras import Model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json, clone_model

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
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights
    loaded_model.load_weights(weights_path)
    # clone model (randomly initializes the weights) trainable
    clone_model = clone_model(loaded_model)

    # get layer index for the last frozen layer
    #  Find the index of the first block3 layer
   
    # get index of last frozen layer
    for index in range(len(loaded_model.layers)):
        if last_frozen_layer in loaded_model.layers[index].name:
            break

    # iterate through the layers freezing weights
    for layer in loaded_model.layers:
        layer.trainable = False
        
    # set output of frozen model to be the last forzen layer
    frozen_layer = loaded_model.get_layer(last_frozen_layer)
    last_output = frozen_layer.output
    x = last_output
    # set input of cloned model to be the last_frozen layer 
    for i in range(index, len(clone_model.layers)):
        clone_model.layers[i].trainable = True
        # connect layer to output
        x = clone_model.layers[i](x)
    # build model from both
    model = Model(loaded_model.input, x)
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