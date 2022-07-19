'''
RESNET-50 MODULE:
'''
# import libraries
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
import time

# src imports
from src.models.model_utils import add_Dense_layers

def build_resnet50(input_shape,
                class_num,
                last_layer,
                dense_nodes,
                lr, 
                dropout, 
                loss, 
                metrics, 
                print_summary=False
                ):
    '''
    Purpose: 
        Initializes and compiles a resnet50 model with pretrained weights from ImageNet up
         to the specified last layer using the passed training parameter arguments. 
        input_shape - tuple - must be 3 channels i.e. (256, 256, 3)
        class_num - int - specified number of classes to set the shape of the output layer (softmax)
        last_layer - str - specified layer to use as the input layer to the dense network.
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
    resnet50_weights = 'models/raw/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    #resnet50_weights = '/Users/garrettmccue/projects/cnn-alzheimers/models/raw/'\
    #                           'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

        # Initialize base model from tensorflow.keras
    # Set the input shape and remove dense layers (top), to match our weights file
    pre_trained_model = ResNet50(input_shape = input_shape,
                                    include_top= False,
                                    weights = None)
    # load pretrained weights
    print(f'Loading resnet50 weights from:\n\t PATH = {resnet50_weights}...')
    pre_trained_model.load_weights(resnet50_weights)


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
    model = add_Dense_layers(pre_trained_model, last_output, dense_nodes, class_num, dropout)
    print('Building Dense layers...')
    print(f'\tAdding Dense layer with {dense_nodes} nodes')
    print(f'\tAdding classification layer for {class_num} classes')

    print('Compiling the model...')
    # -- COMPILE MODEL --
    model.compile(optimizer = Adam(learning_rate=lr),
                  loss = loss,
                  metrics = metrics)

    end = time.time()

    print(f'Model built in: {end-start} seconds ')
    if print_summary:
        model.summary()
    
    return model