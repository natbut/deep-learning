"""
INSERT YOUR NAME HERE
"""

from turtle import width
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
font = {'weight' : 'normal','size'   : 22}
matplotlib.rc('font', **font)
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')



######################################################
# Q1 Implement Init, Forward, and Backward For Layers
######################################################


class SigmoidCrossEntropy:
  
  # Compute the cross entropy loss after sigmoid. The reason they are put into the same layer is because the gradient has a simpler form
  # logits -- batch_size x num_classes set of scores, logits[i,j] is score of class j for batch element i
  # labels -- batch_size x 1 vector of integer label id (0,1) where labels[i] is the label for batch element i
  #
  # TODO: Output should be a positive scalar value equal to the average cross entropy loss after sigmoid
  def forward(self, logits, labels):
    raise Exception('Student error: You haven\'t implemented the forward pass for SigmoidCrossEntropy yet.')


  # TODO: Compute the gradient of the cross entropy loss with respect to the the input logits
  def backward(self):
    raise Exception('Student error: You haven\'t implemented the backward pass for SigmoidCrossEntropy yet.')



class ReLU:

  # TODO: Compute ReLU(input) element-wise
  def forward(self, input):
    raise Exception('Student error: You haven\'t implemented the forward pass for ReLU yet.')
      
  # TODO: Given dL/doutput, return dL/dinput
  def backward(self, grad):
    raise Exception('Student error: You haven\'t implemented the backward pass for ReLU yet.')

  # No parameters so nothing to do during a gradient descent step
  def step(self,step_size, momentum = 0, weight_decay = 0):
    return


class LinearLayer:

  # TODO: Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
  def __init__(self, input_dim, output_dim):
    raise Exception('Student error: You haven\'t implemented the init for LinearLayer yet.')
    
  # TODO: During the forward pass, we simply compute XW+b
  def forward(self, input):
    raise Exception('Student error: You haven\'t implemented the forward pass for LinearLayer yet.')


  # TODO: Backward pass inputs:
  #
  # grad dL/dZ -- For a batch size of n, grad is a (n x output_dim) matrix where 
  #         the i'th row is the gradient of the loss of example i with respect 
  #         to z_i (the output of this layer for example i)

  # Computes and stores:
  #
  # self.grad_weights dL/dW --  A (input_dim x output_dim) matrix storing the gradient
  #                       of the loss with respect to the weights of this layer. 
  #                       This is an summation over the gradient of the loss of
  #                       each example with respect to the weights.
  #
  # self.grad_bias dL/dZ--     A (1 x output_dim) matrix storing the gradient
  #                       of the loss with respect to the bias of this layer. 
  #                       This is an summation over the gradient of the loss of
  #                       each example with respect to the bias.
  
  # Return Value:
  #
  # grad_input dL/dX -- For a batch size of n, grad_input is a (n x input_dim) matrix where
  #               the i'th row is the gradient of the loss of example i with respect 
  #               to x_i (the input of this layer for example i) 

  def backward(self, grad):
    raise Exception('Student error: You haven\'t implemented the backward pass for LinearLayer yet.')
    

  ######################################################
  # Q2 Implement SGD with Weight Decay
  ######################################################  
  def step(self, step_size, momentum = 0.8, weight_decay = 0.0):
  # TODO: Implement the step
    raise Exception('Student error: You haven\'t implemented the step for LinearLayer yet.')






######################################################
# Q4 Implement Evaluation for Monitoring Training
###################################################### 

# TODO: Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
  raise Exception('Student error: You haven\'t implemented the step for evaluate function.')


def main():

  # TODO: Set optimization parameters (NEED TO SUPPLY THESE)
  batch_size = 
  max_epochs = 
  step_size = 

  number_of_layers = 
  width_of_layers = 
  weight_decay = 0.0
  momentum = 0.8


  # Load data
  data = pickle.load(open('cifar_2class_py3.p', 'rb'))
  X_train = data['train_data']
  Y_train = data['train_labels']
  X_test = data['test_data']
  Y_test = data['test_labels']
  
  # Some helpful dimensions
  num_examples, input_dim = X_train.shape
  output_dim = 1 # number of class labels -1 for sigmoid loss


  # Build a network with input feature dimensions, output feature dimension,
  # hidden dimension, and number of layers as specified below. You can edit this as you please.
  net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)

  # Some lists for book-keeping for plotting later
  losses = []
  val_losses = []
  accs = []
  val_accs = []
  

  raise Exception('Student error: You haven\'t implemented the training loop yet.')
  
  # Q2 TODO: For each epoch below max epochs

    # Scramble order of examples

    # for each batch in data:

      # Gather batch

      # Compute forward pass

      # Compute loss

      # Backward loss and networks

      # Take optimizer step

      # Book-keeping for loss / accuracy
  
    # Evaluate performance on test.
    _, tacc = evaluate(net, X_test, Y_test, batch_size)
    print(tacc)

    
    ###############################################################
    # Print some stats about the optimization process after each epoch
    ###############################################################
    # epoch_avg_loss -- average training loss across batches this epoch
    # epoch_avg_acc -- average accuracy across batches this epoch
    # vacc -- testing accuracy this epoch
    ###############################################################
    
    #logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(i,epoch_avg_loss, epoch_avg_acc, vacc*100))

    
  ###############################################################
  # Code for producing output plot requires
  ###############################################################
  # losses -- a list of average loss per batch in training
  # accs -- a list of accuracies per batch in training
  # val_losses -- a list of average testing loss at each epoch
  # val_acc -- a list of testing accuracy at each epoch
  # batch_size -- the batch size
  ################################################################

  # Plot training and testing curves
  fig, ax1 = plt.subplots(figsize=(16,9))
  color = 'tab:red'
  ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
  ax1.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_losses))], val_losses,c="red", label="Val. Loss")
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
  ax1.tick_params(axis='y', labelcolor=color)
  #ax1.set_ylim(-0.01,3)
  
  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  color = 'tab:blue'
  ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
  ax2.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_accs))], val_accs,c="blue", label="Val. Acc.")
  ax2.set_ylabel(" Accuracy", c=color)
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.set_ylim(-0.01,1.01)
  
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  ax1.legend(loc="center")
  ax2.legend(loc="center right")
  plt.show()  



#####################################################
# Feedforward Neural Network Structure
# -- Feel free to edit when tuning
#####################################################

class FeedForwardNeuralNetwork:

  def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
 
    if num_layers == 1:
      self.layers = [LinearLayer(input_dim, output_dim)]
    else:
    # TODO: Please create a network with hidden layers based on the parameters

  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def backward(self, grad):
    for layer in reversed(self.layers):
      grad = layer.backward(grad)

  def step(self, step_size, momentum, weight_decay):
    for layer in self.layers:
      layer.step(step_size, momentum, weight_decay)



def displayExample(x):
  r = x[:1024].reshape(32,32)
  g = x[1024:2048].reshape(32,32)
  b = x[2048:].reshape(32,32)
  
  plt.imshow(np.stack([r,g,b],axis=2))
  plt.axis('off')
  plt.show()


if __name__=="__main__":
  main()