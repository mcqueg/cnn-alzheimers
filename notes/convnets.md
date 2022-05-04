# Convolutional Neural Networks

Articles

- [x] [_Comprehensive Guide to Convolutional Neural Networks_](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)  
- [ ] [_Guide to Convolution Arithmetic for Deep Learning_](https://github.com/vdumoulin/conv_arithmetic)
- [ ] [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)


## ConvNets building blocks

`INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

### Convolution layer

- Kernel
- Stride
- Padding
  - valid padding
  - same padding

### Pooling layer

- Kernel
- Max pooling
- Average pooling

### Fully Connected Layer

- softmax layer

![cnn-example](https://miro.medium.com/max/1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

The preprocessing for a ConvNet is a lot lower than in other classification algorithms. In very basic ConvNets the filters can be hand-engineered, more complex methods allow for the learning of the filter sizes and characteristics.  

ConvNets operate similar to the visual cortex of the brain. Individual neurons respond to stimuli restricted to a specific region. The specific and restricted region is known as the **Receptive Field**.  Receptive fields are created so that they overlap sharing some information between other neighboring receptive fields in order to cover the entire visual area.

We could flatten the image into an array and attempt to classify that way, but it strips the image of vital spatial information amongst pixel location. A ConvNet is a better choice for images because it is able to successfully capture both the **spatial** and **temporal** dependencies within an image.

**Input**  
![cnn-input](https://miro.medium.com/max/1100/1*15yDvGKV47a0nkf5qLKOOQ.png)

Filters, aka Kernels, have a smaller dimension than the input images and are used to reduce the number of pixels into a smaller representation while maintaining the spatial and temporal relationships necessary for classification.
> The role of the ConvNet is to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction  

## Convolution Layer

The kernels are applied to the images in the  convolution step of the architecture with a specified stride in order to cover the entire span, height and width, of each image.

**Convolution operation with a stride length of 2 and a padding of 1**  

![filters-convolutions](https://miro.medium.com/max/790/1*1VJDP6qDY9-ExTuQVEOlVg.gif)

**Convolution of a MxNx3 image matrix with a 3x3x3 kernel**
![3d-kernel](https://miro.medium.com/max/1400/1*ciDgQEjViWLnCbmX-EeSrA.gif)

> The objective of the Convolution Operation is to extract the high-level features such as edges, from the input images.  

Early Convolutional layers in the architecture are important for learning the low level features such as edges, lines, and colors. As more layers are added high level features begin to be learned from the construction of the learned lower level features.

Padding can be added to the borders of the images in order to control the output size of the matrix after the filter is applied. We can add padding to the image to increase its size before the kernel is applied in order to have the output after the convolution to be the same size as the input...this is called **Same Padding**. **Valid padding**, or adding no 0s, will result in the output to be the size of the kernel that was applied to it.

**SAME padding: 5x5x1 image is padded with 0s to create a 6x6x1 image**
![same-padding](https://miro.medium.com/max/790/1*nYf_cUIHFEWU1JXGwnz-Ig.gif)

## Pooling Layer
> It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: A pooling layer with F=3,S=2 (also called overlapping pooling), and more commonly F=2,S=2. Pooling sizes with larger receptive fields are too destructive.

**3x3 pooling over 5x5 convolved feature**  
![pooling](https://miro.medium.com/max/792/1*uoWYsCV5vBU8SHFPAPao-w.gif)

Similar to the convolutional layer, the pooling layer reduces the spatial size of the convolved feature. This is an operation of dimensionality reduction to decrease the needed computational power. It is also can act as a step for feature extraction by extracting the most dominant features while mapping them. Two main types of pooling layers exist

### Max Pooling vs Average Pooling

![max-avg-pool](https://miro.medium.com/max/1192/1*KQIEqhxzICU7thjaQBfPBQ.png)

**Max Pooling**: returns the maximum value from the portion of the matrix that the kernel is covering.

> Max Pooling also performs as a Noise Suppressant. It discards the noisy activations altogether and also performs de-noising along with dimensionality reduction

**Average Pooling**: returns the average of all values covered by the kernel.
> Average Pooling simply performs dimensionality reduction as a noise suppressing mechanism

**Max Pooling is a better performing pooling method than average pooling**

The convolutional layer and pooling later form one distinct layer in the ConvNet
> The Convolutional Layer and the Pooling Layer, together form the i-th layer of a Convolutional Neural Network. Depending on the complexities in the images, the number of such layers may be increased for capturing low-levels details even further, but at the cost of more computational power.

After the model has learned features in the images from the _convolutional and pooling layers_ we can then enable the model to classify the images absed on these learned features in another type of layer, **Fully Connected Layer** (FC Layer)

## Fully Connected Layer

![fc-layer](https://miro.medium.com/max/1400/1*kToStLowjokojIQ7pY2ynQ.jpeg)  
Important in the classification of the images. The matrices from the earlier layers are flattened and fed into the **FC layers**

>Adding a Fully-Connected layer is a (usually) cheap way of learning non-linear combinations of the high-level features as represented by the output of the convolutional layer. The Fully-Connected layer is learning a possibly non-linear function in that space.

The flattening of the images into column vectors and feed forward network predictions that are generated after a full forward iteration allows for the computation of a loss function and the implementation of back propagation in order to allow the model to "learn" over many epochs.

>Over a series of epochs, the model is able to distinguish between dominating and certain low-level features in images and classify them using the Softmax Classification technique.

[GitHub: Recognizing Hand Written Digits using the MNIST Dataset with Tensorflow](https://github.com/ss-is-master-chief/MNIST-Digit.Recognizer-CNNs)
