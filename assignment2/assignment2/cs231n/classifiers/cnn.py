from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    
    '''
    old parameters
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32)
    '''
        
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size
        self.num_filters = num_filters
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.use_batchnorm = use_batchnorm
        self.dtype = dtype
        self.bn_param = {}
        
        #Intialize parameters for conv layers
        C, H, Winput = input_dim
        F = num_filters
        fh = filter_size
        fw = filter_size
        stride = 2
        p = (filter_size - 1)/2
        Hc = (H+2*p - fh) / stride+1
        Wc = (Winput+2*p - fw) / stride+1
        
        W1 = weight_scale*np.random.randn(F, C, fh, fw)
        b1 = np.zeros(F)
        
        #pool layer
        hpool = 2
        wpool = 2
        spool = 2
        Hp = (Hc - hpool) / stride+1
        Wp = (Wc - wpool) / stride+1
        
        Hp, Wp = int(Hp), int(Wp)
        #hidden affine layer
        Hh = hidden_dim
        #original was: W2 = weight_scale*np.random.randn(F*HP*Wp, Hh)
        W2 = weight_scale*np.random.randn(F*int(H/2)*int(Winput/2), Hh)
        #print('init: ',fh,', W2 start: ',F,Hp,' ',Wp)
        b2 = np.zeros(Hh)
        
        #output affine layer
        Ho = num_classes
        W3 = weight_scale*np.random.randn(Hh, Ho)
        b3 = np.zeros(Ho)
        
        self.params.update({'W1':W1, 'W2':W2, 'W3':W3, 'b1':b1, 'b2':b2, 'b3':b3})
        
        if self.use_batchnorm:
            bn_param1 = {'mode':'train', 'running_mean':np.zeros(F), 'running_var':np.zeros(F)}
            gamma1, beta1 = np.ones(F),  np.ones(F)
            bn_param2 = {'mode':'train', 'running_mean':np.zeros(Hh), 'running_var':np.zeros(Hh)}
            gamma2, beta2 = np.ones(Hh), np.ones(Hh)
            
            self.params.update({'bn_param1':bn_param1, 'bn_param2':bn_param2})
            self.params.update({'gamma1':gamma1, 'beta1':beta1, 'gamma2': gamma2, 'beta2':beta2})
        #print(self.filter_size,' ',H,' ',W,' ',self.num_filters)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    
        
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        #filter_size = W1.shape[2]
        filter_size = self.filter_size
        num_filters = self.num_filters
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        
        mode = 'test' if y is None else 'train'
        X = X.astype(self.dtype)
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        if self.use_batchnorm:
            bn_param1, gamma1, beta1 = self.bn_params[
                'bn_param1'], self.params['gamma1'], self.params['beta1']
            bn_param2, gamma2, beta2 = self.bn_params[
                'bn_param2'], self.params['gamma2'], self.params['beta2']
            for key, bn_param in self.bn_params.items():
                bn_param[mode] = mode
        
        N = X.shape[0]
        
        f_size = W1.shape[2]
        conv_param={'stride':1, 'pad':int( (f_size-1)/2 ) }
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        #conv layer
        if self.use_batchnorm:
            beta, gamma, bn_param = beta1, gamma1, bn_param1
            conv_layer, cache_conv_layer = conv_bn_relu_pool_forward(X, W1, b1, conv_param,
                                                                     pool_param, gamma, beta, bn_param)
        else:
            conv_layer, cache_conv_layer = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        
        #print('W1: ',W1.shape,' x: ',X.shape)    
        N, F, Hp, Wp = conv_layer.shape
        
        #hidden layer
        x = conv_layer.reshape((N, F*Hp*Wp))
        #print('x: ',F*Hp*Wp,' w: ',W2.shape,'dims: ',F,' ',Hp,' ',Wp)
        
        if self.use_batchnorm:
            beta, gamma, bn_param = beta2, gamma2, bn_param2
            h_layer, cache_h_layer = affine_bn_relu_forward(x, W2, b2, gamma, beta, bn_param)
        else:
            h_layer, cache_h_layer = affine_relu_forward(x, W2, b2)
        N, Hh = h_layer.shape
        
        #output layer
        scores, cache_scores = affine_forward(h_layer, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        #calculate loss
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss=0
        ls = [W1,W2,W3]
        for w in ls:
            reg_loss += 0.5*self.reg*np.sum(w*w)
        
        loss = data_loss+reg_loss
        
        #backprop in output layer
        dx3, dW3, db3 = affine_backward(dscores, cache_scores)
        dW3 += self.reg*W3
        
        #ordinary(affine) layers
        if self.use_batchnorm:
            dx2, dW2, db2, dgamma2, dbeta2 = affine_bn_relu_backward(
                dx3, cache_hidden_layer)
        else:
            dx2, dW2, db2 = affine_relu_backward(dx3, cache_h_layer)
        dW2 += self.reg*W2
        
        #conv layers
        dx2 = dx2.reshape(N, F, Hp, Wp)
        if self.use_batchnorm:
            dx, dW1, db1, dgamma1, dbeta1 = conv_norm_relu_pool_backward(
                dx2, cache_conv_layer)
            
            grads.update({'beta1': dbeta1,
                          'beta2': dbeta2,
                          'gamma1': dgamma1,
                          'gamma2': dgamma2})
        else:
            dx, dW1, db1 = conv_relu_pool_backward(dx2, cache_conv_layer)

        dW1 += self.reg * W1

        grads.update({'W1': dW1,
                      'b1': db1,
                      'W2': dW2,
                      'b2': db2,
                      'W3': dW3,
                      'b3': db3})
            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads