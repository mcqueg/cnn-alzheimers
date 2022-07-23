'''
INCEPTION-V3 MODULE:
'''
# import libraries
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
import time

# src imports
from src.models.model_utils import add_Dense_layers

def build_inception_v3(input_shape,
                       class_num,
                       last_layer,
                       dense_nodes,
                       dropout, 
                       print_summary=False,
                       ):
    '''
    Purpose: 
        Initializes and compiles an inceptionv3 model with pretrained weights from ImageNet up \
         to the specified last layer using the passed training parameter arguments.
    Parameters:
        input_shape - tuple - must be 3 channels i.e. (256, 256, 3)
        class_num - int - specified number of classes to set the shape of the output layer (softmax)
        last_layer - str - specified layer to use as the input layer to the dense network. \
                            see model.summary() for architecture
        dense_nodes - int - number of nodes to include in dense layer
        lr - float - learning rate for defualt ADAM optimizer
        dropout - float - dropout to be applied after the Dense(1024) layer
        loss - str - specified loss to use
        metrics - list - str - list of metrics to use i.e. metrics = ['accuracy', 'loss']
        print_summary - bool - defualt = False; prints the model summary along with the model
    Returns:
        model - compiled model 
    '''
    start = time.time()

    # -- LOAD WEIGHTS --

    # set the downloaded weights file to a variable
    inception_v3_weights = 'models/raw/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
   
    #inception_v3_weights = '/Users/garrettmccue/projects/cnn-alzheimers/models/raw/\
    #                                inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # Initialize base model from tensorflow.keras
    # Set the input shape and remove dense layers (top), to match our weights file
    pre_trained_model = InceptionV3(input_shape = input_shape,
                                    include_top= False,
                                    weights = None)

    # load pretrained weights
    print(f'Loading Inception-v3 weights from:\n\t PATH = {inception_v3_weights}...')
    pre_trained_model.load_weights(inception_v3_weights)

    # -- FORMATTING PRE-TRAINED MODEL -- 

    # freeze weights of each layer
    for layer in pre_trained_model.layers:
        layer.trainable = False

    print(f'Setting last layer as: {last_layer}')
    # choose last layer from the pretrained model to feed into the dense layers
    last_layer = pre_trained_model.get_layer(last_layer)
    print(f'Last layer output shape: {last_layer.output_shape}')
    last_output = last_layer.output
    
    
    # ---  ADDING DENSE LAYER ---
    # add_Dense_layer() from model_utils.py
    print('Building Dense layers...')
    print(f'\tAdding Dense layer with {dense_nodes} nodes')
    print(f'\tAdding classification layer for {class_num} classes')
    model = add_Dense_layers(pre_trained_model, last_output, dense_nodes, class_num, dropout)    
    
    # # Flatten the output layer to 1 dimension
    # x = layers.Flatten()(last_output)
    # # add a fully connected layer
    # x = layers.Dense(1024, activation='relu')(x)
    # # add the dropout rate of 0.2
    # x = layers.Dropout(dropout)(x)
    # # add a final layer with softmax for classification
    # x = layers.Dense(class_num, activation='softmax')(x)

    # # append the created dense network to the pre_trained_model
    # model = Model(pre_trained_model.input, x)
    
    # # print('Compiling the model...')
    # # # -- COMPILE MODEL --
    # # model.compile(optimizer = Adam(learning_rate=lr),
    # #               loss = loss,
    # #               metrics = metrics)

    end = time.time()

    print(f'Model built in: {end-start} seconds \n')
    if print_summary:
        model.summary()
    
    return model
# -------------------------------------------------------------------------------------------------
def load_inception_v3(weights_path, input_shape, last_layer, dense_nodes, class_num, dropout):
    model = InceptionV3(input_shape = input_shape,
                                    include_top= False,
                                    weights = None)
    last_layer = model.get_layer(last_layer)
    last_output = last_layer.output
    print('Building Dense layers...')
    print(f'\tAdding Dense layer with {dense_nodes} nodes')
    print(f'\tAdding classification layer for {class_num} classes')
    model = add_Dense_layers(model, last_output, dense_nodes, class_num, dropout)  
    # load saved weights from weights path
    model.load_weights(weights_path)
    # freeze weights of each layer
    for layer in model.layers:
        layer.trainable = False

    return model