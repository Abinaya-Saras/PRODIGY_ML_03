## Introduction
In this project, we will build a machine learning model to classify images of cats and dogs. We will use a Support Vector Machine (SVM) for the classification task.

## What is Image classification?
Image classification is the process of categorizing and labeling groups of pixels or vectors within an image based on specific rules. The categorization law can be devised using one or more spectral or textural characteristics.

Different image classification techniques:

## Traditional Techniques

1. **K-Nearest Neighbors (KNN)**: Classifies based on the majority class among k-nearest neighbors using extracted features.
2. **Support Vector Machine (SVM)**: Separates classes with an optimal hyperplane, requiring feature extraction.
3. **Decision Trees and Random Forests**: Decision Trees split data based on features; Random Forests use multiple trees to improve accuracy.

## Deep Learning Techniques

1. **Convolutional Neural Networks (CNNs)**: Learn hierarchical features directly from images, trained end-to-end.
2. **Transfer Learning**: Fine-tunes pre-trained models (e.g., VGG, ResNet) on a specific dataset.
3. **Ensemble Methods**: Combines predictions from multiple models for better accuracy.
4. **Attention Mechanisms and Transformers**: Focus on important parts of images, e.g., Vision Transformer.

## Other Techniques

1. **Bag of Visual Words (BoVW)**: Represents images as histograms of clustered visual features.
2. **Autoencoders: Unsupervised networks that learn compact feature representations for classification.

## Preprocessing the training Data using ImageDataGenerator
One of the methods to prevent overfitting is to have more data. By this, our model will be exposed to more aspects of data and thus will generalize better. To get more data, either you manually collect data or generate data from the existing data by applying some transformations. The latter method is known as Data Augmentation.

**rescale:** rescaling factor. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided.
**shear_range:** This is the shear angle in the counter-clockwise direction in degrees.
**zoom_range:** This zooms the image.
**horizontal_flip:** Randomly flips the input image in the horizontal direction.

## When to use a Sequential model
A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

A Sequential model is **not appropriate** when:

Your model has multiple inputs or multiple outputs
Any of your layers has multiple inputs or multiple outputs
You need to do layer sharing
You want non-linear topology (e.g. a residual connection, a multi-branch model)

## Step 1 - Convolution
This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.

When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers or None, does not include the sample axis), e.g. input_shape=(64, 64, 3) for 64x64 RGB pictures in data_format="channels_last". You can use None when a dimension has variable size.

Arguments Used:

**filters:** Integer, the dimensionality of the output space.
**padding:** one of "valid" or "same". "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
**kernel_size:** An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
**activation:** Activation function to use. If you don't specify anything, no activation is applied.
**strides:** An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width.

## Step 2 - Pooling
Downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (of size defined by pool_size) for each channel of the input.
The window is shifted by strides along each dimension.

## Step 3 - Flattening
Flattens the input. Does not affect the batch size.
Note: If inputs are shaped (batch,) without a feature axis, then flattening adds an extra channel dimension and output shape is (batch, 1).

## Compiling the CNN
**Attributes:**

**optimizer:** String (name of optimizer) or optimizer instance.
**loss:** Loss function.
**metrics:** List of metrics to be evaluated by the model during training and testing.

## Training the CNN on the Training set and evaluating it on the Test set
**Attributes:**

**x:** Input data
**validation_data:** Data on which to evaluate the loss and any model metrics at the end of each epoch.
**epochs:** Integer. Number of epochs to train the model.

## Conclusion
This was Just a Introduction to Image Classification. In the upcomming Noteboook i will work on more complex real world problems.
