from tensorflow.keras import layers
from tensorflow.keras import Model

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
        dense_nodes - int- determines the number of nodes to include in the first dense layer
        class_num-int- determines the number of output nodes are needed in the classification layer.
                    Note: this should be set to the number of predicted classes.
        droput-float- specified dropout to apply to the dense layer. 
    '''
    print('Building Dense layers...')
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # add a fully connected layer
    x = layers.Dense(dense_nodes, activation='relu')(x)
    # add the dropout rate of 0.2
    x = layers.Dropout(dropout)(x)
    # add a final layer with softmax for classification
    x = layers.Dense(class_num, activation='softmax')(x)

    # append the created dense network to the pre_trained_model
    dense_model = Model(model.input, x)


    return dense_model