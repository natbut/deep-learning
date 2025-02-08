"""
Nathan Butler
butlnath@oregonstate.edu
"""

import pickle
from turtle import width

import kagglehub
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
  
  # Output should be a positive scalar value equal to the average cross entropy loss after sigmoid
  def forward(self, logits, labels):
    self.labels = labels
    self.sigmoid_output = 1 / (1 + np.exp(-logits))
    self.loss = -np.mean(labels * np.log(self.sigmoid_output + 1e-9) + (1 - labels) * np.log(1 - self.sigmoid_output + 1e-9))
    return self.loss

  # Compute the gradient of the cross entropy loss with respect to the input logits
  def backward(self):
    # dL/dz = dL/dy * dy/dz
    grad = (self.sigmoid_output - self.labels)
    return grad


class ReLU:

  # Compute ReLU(input) element-wise
  def forward(self, input):
    self.input = input
    return np.maximum(0, input)
      
  # Given dL/doutput, return dL/dinput
  def backward(self, grad):
    grad_input = grad.copy()
    grad_input[self.input <= 0] = 0
    return grad_input

  # No parameters so nothing to do during a gradient descent step
  def step(self, step_size, momentum=0, weight_decay=0):
    return


class LinearLayer:

  # Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
  def __init__(self, input_dim, output_dim):
    # self.weights = abs(np.random.randn(input_dim, output_dim)*0.01)
    self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
    self.bias = np.zeros((1, output_dim))
    self.input = None
    self.grad_weights = None
    self.grad_bias = None
    self.velocity_weights = np.zeros_like(self.weights)
    self.velocity_bias = np.zeros_like(self.bias)
    
  # During the forward pass, we simply compute XW+b
  def forward(self, input):
    self.input = input
    # print("Lin output: ", np.dot(input, self.weights) + self.bias)
    return np.dot(input, self.weights) + self.bias

  # Backward pass inputs:
  #
  # grad dL/dZ -- For a batch size of n, grad is a (n x output_dim) matrix where the i'th row is the gradient of the loss of example i with respect to z_i (the output of this layer for example i)

  # Computes and stores:
  #
  # self.grad_weights dL/dW --  A (input_dim x output_dim) matrix storing the gradient of the loss with respect to the weights of this layer. This is an summation over the gradient of the loss of each example with respect to the weights.
  #
  # self.grad_bias dL/dZ--     A (1 x output_dim) matrix storing the gradient of the loss with respect to the bias of this layer. This is an summation over the gradient of the loss of each example with respect to the bias.
  
  # Return Value:
  #
  # grad_input dL/dX -- For a batch size of n, grad_input is a (n x input_dim) matrix where the i'th row is the gradient of the loss of example i with respect to x_i (the input of this layer for example i) 

  def backward(self, grad):
    batch_size = self.input.shape[0]
    self.grad_weights = np.dot(self.input.T, grad) / batch_size # TODO batch normalization?
    self.grad_bias = np.sum(grad, axis=0, keepdims=True) / batch_size # TODO batch normalization?
    grad_input = np.dot(grad, self.weights.T)
    return grad_input

  ######################################################
  # Q2 Implement SGD with Weight Decay
  ######################################################  
  def step(self, step_size, momentum = 0.8, weight_decay = 0.0):
    self.velocity_weights = momentum * self.velocity_weights - step_size * (self.grad_weights + weight_decay * self.weights)
    self.velocity_bias = momentum * self.velocity_bias - step_size * (self.grad_bias + weight_decay * self.bias)
    self.weights += self.velocity_weights
    self.bias += self.velocity_bias
    # print(f"Weights: {self.weights} \nBias: {self.bias}")


######################################################
# Q4 Implement Evaluation for Monitoring Training
###################################################### 

# Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
  num_examples = X_val.shape[0]
  losses = []
  accs = []
  loss_fn = SigmoidCrossEntropy()
  for i in range(0, num_examples, batch_size):
    X_batch = X_val[i:i+batch_size]
    Y_batch = Y_val[i:i+batch_size]

    logits = model.forward(X_batch)
    loss = loss_fn.forward(logits, Y_batch)
    losses.append(loss)

    predictions = (logits > 0.5).astype(int)
    accs.append(np.sum(predictions == Y_batch)/len(predictions))

  avg_loss = np.mean(losses)
  avg_acc = np.mean(accs)

  return avg_loss, avg_acc


def main():  

  # NOTE: Set optimization parameters (NEED TO SUPPLY THESE)
  batch_size = 256
  max_epochs = 250
  step_size = 0.001

  number_of_layers = 5
  width_of_layers = 256
  weight_decay = 0.00001
  momentum = 0.8 #0.8

  # Load data
  data = pickle.load(open('cifar_2class_py3.p', 'rb'))
  # data = pickle.load(open('xor_Dataset.csv', 'rb'))
  X_train = data['train_data'] / 255.0
  Y_train = data['train_labels']
  X_test = data['test_data'] / 255.0
  Y_test = data['test_labels']
  
  # Some helpful dimensions
  num_examples, input_dim = X_train.shape
  output_dim = 1 # number of class labels -1 for sigmoid loss

  # Build a network with input feature dimensions, output feature dimension,
  # hidden dimension, and number of layers as specified below. You can edit this as you please.
  net = FeedForwardNeuralNetwork(input_dim,
                                 output_dim,
                                 width_of_layers,
                                 number_of_layers
                                 )

  # Some lists for book-keeping for plotting later
  losses = []
  val_losses = []
  accs = []
  val_accs = []
  
  loss_fn = SigmoidCrossEntropy()
  # Q2 For each epoch below max epochs
  for ep in range(max_epochs):
    
    # Scramble order of examples
    indices = np.random.permutation(X_train.shape[0])
    epoch_loss = 0
    epoch_correct = 0
    
    num_batches = num_examples // batch_size
    
    # for each batch in data:
    for batch in range(num_batches):

      # Gather batch
      X_batch = X_train[indices[batch*batch_size:(batch+1)*batch_size]]
      Y_batch = Y_train[indices[batch*batch_size:(batch+1)*batch_size]]
      
      # Compute forward pass
      logits = net.forward(X_batch)
      
      # Compute loss
      loss = loss_fn.forward(logits, Y_batch)
    
      # Backward loss and networks
      grad = loss_fn.backward()
      net.backward(grad)

      # Take optimizer step
      net.step(step_size, momentum, weight_decay)

      # Book-keeping for Training monitoring
      losses.append(loss)
      epoch_loss += loss
      predictions = (logits > 0.5).astype(int)
      epoch_correct += np.sum(predictions == Y_batch)
      accs.append(np.mean(predictions == Y_batch))
  
    # Evaluate performance on test.
    tloss, tacc = evaluate(net, X_test, Y_test, batch_size)
    val_losses.append(tloss)
    val_accs.append(tacc)

    # Calculate epoch averages
    epoch_avg_loss = epoch_loss / num_batches
    epoch_avg_acc = epoch_correct / num_examples

    ###############################################################
    # Print some stats about the optimization process after each epoch
    ###############################################################
    # epoch_avg_loss -- average training loss across batches this epoch
    # epoch_avg_acc -- average accuracy across batches this epoch
    # vacc -- testing accuracy this epoch
    ###############################################################
    logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}% ".format(ep, epoch_avg_loss, epoch_avg_acc*100, tacc*100))
    # print("Weights:", net.layers[-1].weights)
    # print("Bias:", net.layers[-1].bias)
    # print("Grad:", grad)
    # print("Logit:", logits[0], " Sigmoid: ", loss_fn.sigmoid_output[0], " Label: ", Y_batch[0], "Loss:", loss, "Grad:", grad[0])
    
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
    # NOTE: Please create a network with hidden layers based on the parameters
      self.layers = [LinearLayer(input_dim, hidden_dim)]
      hidden1 = hidden_dim
      for _ in range(num_layers-2):
        hidden2 = hidden1#//2
        self.layers.append(ReLU())
        self.layers.append(LinearLayer(hidden1, hidden2))
        hidden1 = hidden2
      self.layers.append(ReLU())
      self.layers.append(LinearLayer(hidden2, output_dim))

  def forward(self, X):
    # print(f"Input: {X}")
    for layer in self.layers:
      X = layer.forward(X)
      # print(f"Layer: {layer} Output: {X}")
    #   if layer == self.layers[-2]:
    #     print("Final ReLU", X)
    # print("Logits:", X)
    return X

  def backward(self, grad):
    # print("\n")
    for layer in reversed(self.layers):
      # print("Layer", layer, "Avg grad: ", np.mean(grad))
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