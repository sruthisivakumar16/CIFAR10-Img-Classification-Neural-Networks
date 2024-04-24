# CIFAR10-Img-Classification-Neural-Networks

The goal of this project is to develop and train a custom neural network model that can classify images from the CIFAR-10 dataset with the highest possible accuracy. This involves implementing a basic architecture, experimenting with various training techniques and hyperparameter adjustments, and making improvements to the architecture to enhance the model's performance on unseen test data. The project aims not only to achieve a high classification accuracy but also to explore and document the effectiveness of different strategies in machine learning for image recognition tasks.

(Done as a part of my uni coursework)

## Basic Architecture 
<b>Overview:</b> Uses a sequence of intermediate blocks followed by an output block.

### Intermediate Blocks:
- Each block processes the same input image through multiple independent convolutional layers.
- Outputs of these layers are combined using a weighted sum, where weights are derived from a fully connected layer that computes these based on the average channel values of the input image.

### Output Block:
- Receives the combined output of the last intermediate block.
- Uses global average pooling to reduce this output to a channel-wise vector.
- This vector feeds into one or more fully connected layers to produce the classification logits.

## Improved Architecture

### Intermediate Blocks:
- Three convolutional layers with batch normalization and sigmoid activation, maintaining the size due to padding.
- A dropout layer with a 0.1 rate to combat overfitting.
- Use of MaxPool2d layers to progressively reduce the spatial dimensions of the tensor.
- Increasing complexity through additional intermediate blocks with more convolutional layers and higher output channels (64, 128, 256).

### Output Block:
- Processes a flattened tensor from the last pooling layer into a fully connected layer.
- Uses softmax to output a probability distribution over the classes.

## ML Workflow

### Data Handling:
Utilizes PyTorch's DataLoader to manage data efficiently, ensuring batches are appropriately sized and data is shuffled for training variability.

### Training Techniques:
- Data augmentation strategies such as random cropping, flipping, and rotation to increase dataset diversity and improve the model's ability to generalize across different image orientations and compositions
- Hyperparameter tuning focusing on optimising model parameters including dropout rates to prevent overfitting, applying batch normalization for more stable training, and fine-tuning the learning rate for optimal convergence.

### Testing and Evaluation:
- Training over 50 epochs, closely monitoring both training and testing accuracies to evaluate the model's performance and making necessary adjustments to parameters like the learning rate based on ongoing performance metrics.
- Rigorous evaluation approach to determine the model's effectiveness, focusing on accuracy metrics and analyzing trends in loss and learning stability across epochs.

### Results:
Achieved a maximum training accuracy of 90.36% and a testing accuracy of 87.33%, providing a detailed analysis of the performance metrics throughout the training epochs, which illustrates the model's learning progression and its capacity to generalize from training to unseen data.
