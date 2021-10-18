import torch
from torch import nn
import numpy as np


class ESN(nn.Module):
    
    def __init__(self, states_dim, output_dim, proba, sigma, tau):
        super(ESN, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.proba = proba  # Recurrent weights probability of connection
        self.sigma = sigma  # Recurrent weights aplitude
        self.tau = tau  # Time constant
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) * sigma
        self.connect = torch.rand(self.states_dim, self.states_dim) < proba
        self.w_r *= self.connect
        self.b_r = torch.randn(self.states_dim)/10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h = None
        
    def forward(self, x, h_init, store=True):
        """
        Forward pass through the network. The target trajectory x is only provided to compute
          prediction error, and does not influence the prediction (there is no feedback).
        Parameters:
        - x : target sequence, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store as attributes the different activations, should be set to
            True if learning is going to be performed on the target sequence, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        h = h_init
                
        for t in range(seq_len):
            
            # Compute h
            h = (1-1/self.tau) * h + (1/self.tau) * (
                torch.mm(
                    torch.tanh(h),
                    self.w_r.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                hs[t] = h.detach()

            # Compute x_pred according to h
            x_pred =  torch.mm(torch.tanh(h), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()

            # Compute the error on the output level
            error = x_pred - x[t]
            errors[t] = error
            
        if store:
            self.h = hs
            self.x_pred = x_preds
            self.error = errors
            
        return errors

    def learn(self, lr):
        """
        The implementation provided here uses the sequence length as batch dimension to speed up computations
          Still the learning rule does not use any non-local or past/future information and thus can be applied
          in parallel with the forward computations of the RNN
        Parameters:
        - lr: learning rate of the output_weights, double
        """
        
        seq_len, batch_size, _ = self.x_pred.shape
        
        delta_w_o = torch.mm(
            self.error.reshape(seq_len*batch_size, self.output_dim).T, 
            torch.tanh(self.h_prior.reshape(seq_len*batch_size, self.states_dim))
        )
        self.w_o -= lr * delta_w_o
        
        delta_b_o = torch.sum(self.error.reshape(seq_len*batch_size, self.output_dim), axis=0)
        self.b_o -= lr * delta_b_o
        
####################################################################################################################################

class OutRS(nn.Module):
    
    def __init__(self, states_dim, output_dim, proba, sigma, tau):
        super(OutRS, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.proba = proba  # Recurrent weights probability of connection
        self.sigma = sigma  # Recurrent weights aplitude
        self.tau = tau  # Time constant
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) * sigma
        self.connect = torch.rand(self.states_dim, self.states_dim) < proba
        self.w_r *= self.connect
        self.b_r = torch.randn(self.states_dim)/10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h = None
        
        # Reset learning
        self.reset_learning()
        
    def forward(self, x, h_init, store=True):
        """
        Forward pass through the network. The target trajectory x is only provided to compute
          prediction error, and does not influence the prediction (there is no feedback).
        Parameters:
        - x : target sequence, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store as attributes the different activations, should be set to
            True if learning is going to be performed on the target sequence, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        h = h_init
                
        for t in range(seq_len):
            
            # Compute h
            h = (1-1/self.tau) * h + (1/self.tau) * (
                torch.mm(
                    torch.tanh(h),
                    self.w_r.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                hs[t] = h.detach()

            # Compute x_pred according to h
            x_pred =  torch.mm(torch.tanh(h), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()

            # Compute the error on the output level
            error = x_pred - x[t]
            errors[t] = error
            
        if store:
            self.h = hs
            self.x_pred = x_preds
            self.error = errors
            
        return errors

    def reset_learning(self):
        self.past_error=float('inf')
        self.noise = torch.zeros_like(self.w_o)
    
    def learn(self, sigma_r):
        """
        The implementation provided here uses the sequence length as batch dimension to speed up computations
          Still the learning rule does not use any non-local or past/future information and thus can be applied
          in parallel with the forward computations of the RNN
        Parameters:
        - sigma_r: standard deviation of the random search, double
        """
        
        # Whether to save the last applied noise
        new_error = torch.mean(torch.sqrt(torch.sum(self.error**2, axis=2)))
        if self.past_error > new_error:
            self.past_error = new_error
        else:
            self.w_o -= self.noise
            
        # Apply new noise on the output weights
        self.noise = torch.rand_like(self.w_o) * sigma_r
        self.w_o += self.noise
        
####################################################################################################################################

class Conceptors(nn.Module):
    
    def __init__(self, states_dim, output_dim, proba, sigma, tau, alpha):
        super(Conceptors, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.proba = proba  # Recurrent weights probability of connection
        self.sigma = sigma  # Recurrent weights aplitude
        self.tau = tau  # Time constant
        self.alpha = alpha  # Aperture coefficient
        
        # Initialize the Covariance matrix and Conceptor matrix
        self.T = torch.zeros(self.states_dim, self.states_dim)  # Current task covariance matrix
        self.i = 0  # Counter for the number of training iterations on the current task
        self.R = torch.zeros(self.states_dim, self.states_dim)  # All past tasks covariance matrix
        self.k = 0  # Counter for the number of all past tasks
        self.C = torch.zeros(self.states_dim, self.states_dim)  # All past tasks Conceptor matrix
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) * sigma
        self.connect = torch.rand(self.states_dim, self.states_dim) < proba
        self.w_r *= self.connect
        self.b_r = torch.randn(self.states_dim)/10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h = None
        
    def forward(self, x, h_init, store=True):
        """
        Forward pass through the network. The target trajectory x is only provided to compute
          prediction error, and does not influence the prediction (there is no feedback).
        Parameters:
        - x : target sequence, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store as attributes the different activations, should be set to
            True if learning is going to be performed on the target sequence, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        h = h_init
                
        for t in range(seq_len):
            
            # Compute h
            h = (1-1/self.tau) * h + (1/self.tau) * (
                torch.mm(
                    torch.tanh(h),
                    self.w_r.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                hs[t] = h.detach()

            # Compute x_pred according to h
            x_pred =  torch.mm(torch.tanh(h), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()

            # Compute the error on the output level
            error = x_pred - x[t]
            errors[t] = error
            
        if store:
            self.h = hs
            self.x_pred = x_preds
            self.error = errors      
        
        return errors


    def reset_covariance(self):
        """
        Resets the current task covariance matrix, typically called at the beginning of a new task
        """
        self.T = torch.zeros(self.states_dim, self.states_dim)
        self.i = 0  # Counter for the number of training iterations on the current task


    def switch_task(self):
        """
        Function that updates the past tasks covariance matrix and Conceptor matrix, and resets the current task covariance matrix
        """
        # Increment the counter
        self.k += 1

        # Update the all past tasks covariance matrix
        self.R = (1-1/self.k) * self.R + (1/self.k) * self.T

        # Update the all past tasks Conceptor
        self.C = torch.mm(self.R, torch.Tensor(np.linalg.inv(self.R.numpy() + 1/self.alpha**2 * np.identity(self.states_dim))))

        # Reset the current task covariance matrix
        self.reset_covariance()


    def learn(self, lr):
            """
            The implementation provided here uses the sequence length as batch dimension to speed up computations
              Still the learning rule does not use any non-local or past/future information and thus can be applied
              in parallel with the forward computations of the RNN
            Parameters:
            - lr: learning rate of the output_weights, double
            """

            seq_len, batch_size, _ = self.x_pred.shape

            hs = torch.tanh(self.h.reshape(seq_len*batch_size, self.states_dim))

            # Update the covariance matrix using the hidden state trajectory
            self.i += 1
            self.T = (1-1/self.i) * self.T + (1/self.i) * torch.mm(hs.T, hs)

            # Update the output weights using the Conceptors matrix to project the hidden states into a subspace 
            # where learning minimally interferes with previously acquired skills/knowledge
            delta_w_o = torch.mm(
                self.error.reshape(seq_len*batch_size, self.output_dim).T, 
                torch.mm(hs, torch.eye(self.states_dim) - self.C)
            )
            self.w_o -= lr * delta_w_o
    
####################################################################################################################################
    
class PC_RNN_V(nn.Module):
    
    def __init__(self, states_dim, output_dim, tau_h, alpha_x):
        super(PC_RNN_V, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.tau_h = tau_h  # Time constant of the network
        self.alpha_x = alpha_x  # Update rate coefficient of the hidden state layer
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_r = torch.randn(self.states_dim)/10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        
    def forward(self, x, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store as attributes the different activations, should be set to
            True if learning is going to be performed on the target sequence, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h_prior = (1-1/self.tau_h) * old_h_post + (1/self.tau_h) * (
                torch.mm(
                    torch.tanh(old_h_post),
                    self.w_r.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                h_priors[t] = h_prior.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
          
            # Compute the error on the output level
            error = x_pred - x[t]
            errors[t] = error
                
            # Bottom-up pass
            if self.alpha_x > 0:

                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*(1-torch.tanh(h_prior)**2)*torch.mm(error, self.w_o)
                if store:
                    h_posts[t] = h_post.detach()

                # Compute the error on the hidden state level
                error_h = h_prior - h_post
                if store:
                    error_hs[t] = error_h.detach()
                    
                old_h_post = h_post
            
            else:
                old_h_post = h_prior
    
        
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.error_h = error_hs
            self.h_prior = h_priors
            self.h_post = h_posts
        return errors
    
    def learn(self, lr, lr_r):
        """
        The implementation provided here uses the sequence length as batch dimension to speed up computations
          Still the learning rule does not use any non-local or past/future information and thus can be applied
          in parallel with the forward computations of the RNN
        Parameters:
        - lr: learning rate of the output_weights, double
        - lr_r: learning rate of the recurrent weights, double
        """
        
        seq_len, batch_size, _ = self.x_pred.shape
        
        delta_w_o = torch.mm(
            self.error.reshape(seq_len*batch_size, self.output_dim).T, 
            torch.tanh(self.h_prior.reshape(seq_len*batch_size, self.states_dim))
        )
        self.w_o -= lr * delta_w_o
        
        delta_b_o = torch.sum(self.error.reshape(seq_len*batch_size, self.output_dim), axis=0)
        self.b_o -= lr * delta_b_o
        
        delta_w_r = torch.mm(
            self.error_h[1:].reshape((seq_len-1)*batch_size, self.states_dim).T,
            torch.tanh(self.h_post[:-1].reshape((seq_len-1)*batch_size, self.states_dim))
        )
        self.w_r -= lr_r * delta_w_r
            
        delta_b_r = torch.sum(self.error_h.reshape(seq_len*batch_size, self.states_dim), axis=0)
        self.b_r -= lr_r * delta_b_r
    
####################################################################################################################################
    
class P_TNCN(nn.Module):
    
    def __init__(self, states_dim, output_dim, tau, alpha, beta):
        super(P_TNCN, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.tau = tau  # Time constant of the network
        self.alpha = alpha  # Importance of the sparsity constraint
        self.beta = beta  # Importance of the inference update

        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_r = torch.randn(self.states_dim)/10
        
        # Feedback weights initialization
        self.w_f = torch.randn(self.states_dim, self.output_dim) / self.output_dim
        
        # Bottom-up weights initialization
        self.w_i = torch.randn(self.states_dim, self.output_dim) / self.output_dim
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x = None
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        self.error_h = None

    def forward(self, x, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store as attributes the different activations, should be set to
            True if learning is going to be performed on the target sequence, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        old_x = torch.zeros(batch_size, self.output_dim)
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h_prior = (1-1/self.tau) * old_h_post + (1/self.tau) * (
                torch.mm(
                    torch.tanh(old_h_post),
                    self.w_r.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1) \
                + torch.mm(old_x, self.w_i.T)
            )
            if store:
                h_priors[t] = h_prior.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
          
            # Compute the error on the output level
            error = x_pred - x[t]
            errors[t] = error
                
            # Bottom-up pass
            if self.beta > 0:
                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha * torch.sign(h_prior) - self.beta*torch.mm(error, self.w_f.T)
                x_old = x[t]
            else:
                h_post = h_prior - self.alpha * torch.sign(h_prior)
                x_old = x_pred
                
            if store:
                h_posts[t] = h_post.detach()

            # Compute the error on the hidden state level
            error_h = h_prior - h_post
            if store:
                error_hs[t] = error_h.detach()

            old_h_post = h_post   
        
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.error_h = error_hs
            self.h_prior = h_priors
            self.h_post = h_posts
        return errors
    
    def learn(self, lr, lr_f, lr_i, lr_r):
        """
        The implementation provided here uses the sequence length as batch dimension to speed up computations
          Still the learning rule does not use any non-local or past/future information and thus can be applied
          in parallel with the forward computations of the RNN
        Parameters:
        - lr: learning rate of the output_weights, double
        - lr_f: learning rate of the feedback weights, double
        - lr_i: learning rate of the bottom-up weights, double
        - lr_r: learning rate of the recurrent weights, double
        """
        
        seq_len, batch_size, _ = self.x_pred.shape
        
        delta_w_r = torch.mm(
            self.error_h[1:].reshape((seq_len-1)*batch_size, self.states_dim).T, 
            torch.tanh(self.h_post[:-1].reshape((seq_len-1)*batch_size, self.states_dim))
        )
        self.w_r -= lr_r * delta_w_r

        delta_w_i = torch.mm(
            self.error_h[1:].reshape((seq_len-1)*batch_size, self.states_dim).T, 
            self.x_pred[:-1].reshape((seq_len-1)*batch_size, self.output_dim)
        )
        self.w_i -= lr_i * delta_w_i
 
        delta_w_o = torch.mm(
            self.error.reshape(seq_len*batch_size, self.output_dim).T, 
            torch.tanh(self.h_prior.reshape(seq_len*batch_size, self.states_dim))
        )
        self.w_o -= lr * delta_w_o
        
        delta_b_o = torch.sum(self.error.reshape(seq_len*batch_size, self.output_dim), axis=0)
        self.b_o -= lr * delta_b_o
        
        delta_w_f = torch.mm(
            (self.error_h[1:] - self.error_h[:-1]).reshape((seq_len-1)*batch_size, self.states_dim).T,
            self.error[1:].reshape((seq_len-1)*batch_size, self.output_dim)
        )
        self.w_f -= lr_f * delta_w_f

        delta_w_r = torch.mm(
            self.error_h[1:].reshape((seq_len-1)*batch_size, self.states_dim).T,
            torch.tanh(self.h_post[:-1].reshape((seq_len-1)*batch_size, self.states_dim))
        )
        self.w_r -= lr_r * delta_w_r
            
        delta_b_r = torch.sum(self.error_h.reshape(seq_len*batch_size, self.states_dim), axis=0)
        self.b_r -= lr_r * delta_b_r
    
####################################################################################################################################
    
class PC_RNN_Hebb(nn.Module):
    
    def __init__(self, states_dim, output_dim, tau_h, alpha_x):
        super(PC_RNN_Hebb, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.tau_h = tau_h  # Time constant of the network
        self.alpha_x = alpha_x  # Update rate coefficient of the hidden state layer
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_r = torch.randn(self.states_dim)/10
        
        # Feedback weights initialization
        self.w_f = torch.randn(self.states_dim, self.output_dim) / self.output_dim
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        
    def forward(self, x, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store as attributes the different activations, should be set to
            True if learning is going to be performed on the target sequence, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h_prior = (1-1/self.tau_h) * old_h_post + (1/self.tau_h) * (
                torch.mm(
                    torch.tanh(old_h_post),
                    self.w_r.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                h_priors[t] = h_prior.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
          
            # Compute the error on the output level
            error = x_pred - x[t]
            errors[t] = error
                
            # Bottom-up pass
            if self.alpha_x > 0:

                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*torch.mm(error, self.w_f.T)
                if store:
                    h_posts[t] = h_post.detach()

                # Compute the error on the hidden state level
                error_h = h_prior - h_post
                if store:
                    error_hs[t] = error_h.detach()
                    
                old_h_post = h_post
            
            else:
                old_h_post = h_prior
    
        
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.error_h = error_hs
            self.h_prior = h_priors
            self.h_post = h_posts
        return errors
    
    def learn(self, lr, lr_f, lr_r):
        """
        The implementation provided here uses the sequence length as batch dimension to speed up computations
          Still the learning rule does not use any non-local or past/future information and thus can be applied
          in parallel with the forward computations of the RNN
        Parameters:
        - lr: learning rate of the output_weights, double
        - lr_f: learning rate of the feedback weights, double
        - lr_r: learning rate of the recurrent weights, double
        """
        
        seq_len, batch_size, _ = self.x_pred.shape
        
        delta_w_r = torch.mm(
            self.error_h[1:].reshape((seq_len-1)*batch_size, self.states_dim).T, 
            torch.tanh(self.h_post[:-1].reshape((seq_len-1)*batch_size, self.states_dim))
        )
        self.w_r -= lr_r * delta_w_r
 
        delta_w_o = torch.mm(
            self.error.reshape(seq_len*batch_size, self.output_dim).T, 
            torch.tanh(self.h_prior.reshape(seq_len*batch_size, self.states_dim))
        )
        self.w_o -= lr * delta_w_o
        
        delta_b_o = torch.sum(self.error.reshape(seq_len*batch_size, self.output_dim), axis=0)
        self.b_o -= lr * delta_b_o
        
        delta_w_f = delta_w_o.T
        self.w_f -= lr_f * delta_w_f

        delta_w_r = torch.mm(
            self.error_h[1:].reshape((seq_len-1)*batch_size, self.states_dim).T,
            torch.tanh(self.h_post[:-1].reshape((seq_len-1)*batch_size, self.states_dim))
        )
        self.w_r -= lr_r * delta_w_r
            
        delta_b_r = torch.sum(self.error_h.reshape(seq_len*batch_size, self.states_dim), axis=0)
        self.b_r -= lr_r * delta_b_r

####################################################################################################################################
    
class RecRS(nn.Module):
    
    def __init__(self, states_dim, output_dim, tau_h):
        super(RecRS, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.tau_h = tau_h  # Time constant of the network
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_r = torch.randn(self.states_dim)/10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h = None

        # Reset learning
        self.reset_learning()
        
    def forward(self, x, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store as attributes the different activations, should be set to
            True if learning is going to be performed on the target sequence, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            hs = torch.zeros(seq_len, batch_size, self.states_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h = h_init
        
        for t in range(seq_len):
            
            # Compute h_prior according to past h_post
            h = (1-1/self.tau_h) * old_h + (1/self.tau_h) * (
                torch.mm(
                    torch.tanh(old_h),
                    self.w_r.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                hs[t] = h.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
          
            # Compute the error on the output level
            error = x_pred - x[t]
            errors[t] = error
                
            old_h = h
        
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.h = hs
            
        return errors
        
    def reset_learning(self):
        self.past_error=float('inf')
        self.noise = torch.zeros_like(self.w_r)
        
    def learn(self, lr, sigma_r):
        """
        The implementation provided here uses the sequence length as batch dimension to speed up computations
          Still the learning rule does not use any non-local or past/future information and thus can be applied
          in parallel with the forward computations of the RNN
        Parameters:
        - lr: learning rate of the output_weights, double
        - sigma_r: standard deviation of the recurrent weights random search, double
        """
        
        seq_len, batch_size, _ = self.x_pred.shape
        
        delta_w_o = torch.mm(
            self.error.reshape(seq_len*batch_size, self.output_dim).T, 
            torch.tanh(self.h.reshape(seq_len*batch_size, self.states_dim))
        )
        self.w_o -= lr * delta_w_o
        
        delta_b_o = torch.sum(self.error.reshape(seq_len*batch_size, self.output_dim), axis=0)
        self.b_o -= lr * delta_b_o
                
        # Whether to save the last applied noise
        new_error = torch.mean(torch.sqrt(torch.sum(self.error**2, axis=2)))
        if self.past_error > new_error:
            self.past_error = new_error
        else:
            self.w_r -= self.noise
            
        # Apply new noise on the output weights
        self.noise = torch.rand_like(self.w_r) * sigma_r
        self.w_r += self.noise

####################################################################################################################################

class PC_RNN_HC_A(nn.Module):
    
    def __init__(self, causes_dim, states_dim, output_dim, tau_h, alpha_x, alpha_h):
        super(PC_RNN_HC_A, self).__init__()
        
        self.causes_dim = causes_dim
        self.output_dim = output_dim
        self.states_dim = states_dim
        self.tau_h = tau_h  # Time constant of the network
        self.alpha_x = alpha_x  # Update rate coefficient of the hidden state layer
        self.alpha_h = alpha_h  # Update rate coefficient of the hidden causes (input) layer
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim) / 10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_r = torch.randn(self.states_dim) / 10

        # Input weights initialization
        self.w_c = torch.randn(self.states_dim, self.causes_dim) / self.causes_dim
        
        # Predictions, states and errors are temporarily stored for batch learning
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        self.c = None
        
    def forward(self, x, c_init, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - c_init : initial hidden causes (input), Tensor of shape (batch_size, states_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store as attributes the different activations, should be set to
            True if learning is going to be performed on the target sequence, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            cs = torch.zeros(seq_len, batch_size, self.causes_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        c = c_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h_prior = (1-1/self.tau_h) * old_h_post + (1/self.tau_h) * (
                torch.mm(
                    torch.tanh(old_h_post),
                    self.w_r.T
                ) + \
                torch.mm(
                    c,
                    self.w_c.T
                ) + \
                self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                h_priors[t] = h_prior.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
            
            # Compute the error on the output level
            error = x_pred - x[t]
            errors[t] = error
            
            # Bottom-up pass
            if self.alpha_x > 0:

                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*(1-torch.tanh(h_prior)**2)*torch.mm(error, self.w_o)
                if store:
                    h_posts[t] = h_post.detach()

                # Compute the error on the hidden state level
                error_h = h_prior - h_post
                if store:
                    error_hs[t] = error_h.detach()

                # Infer c according to its past value and the error on the hidden state level
                c = c - self.alpha_h*torch.mm(error_h, self.w_c)
                if store:
                    cs[t] = c
                             
                old_h_post = h_post
            
            else:
                old_h_post = h_prior
                
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.error_h = error_hs
            self.h_prior = h_priors
            self.h_post = h_posts
            self.c = cs
                             
        return errors
    
    def learn(self, lr, lr_r, lr_c):
        """
        The implementation provided here uses the sequence length as batch dimension to speed up computations
          Still the learning rule does not use any non-local or past/future information and thus can be applied
          in parallel with the forward computations of the RNN
        Parameters:
        - lr: learning rate of the output_weights, double
        - lr_r: learning rate of the recurrent weights, double
        - lr_c: learning rate of the input weights, double
        """
                
        seq_len, batch_size, _ = self.x_pred.shape
        
        delta_w_o = torch.mm(
            self.error.reshape(seq_len*batch_size, self.output_dim).T, 
            torch.tanh(self.h_prior.reshape(seq_len*batch_size, self.states_dim))
        )
        self.w_o -= lr * delta_w_o

        delta_b_o = torch.sum(self.error.reshape(seq_len*batch_size, self.output_dim), axis=0)
        self.b_o -= lr * delta_b_o
        
        delta_w_c = torch.mm(
            self.error_h.reshape(seq_len*batch_size, self.states_dim).T,
            self.c.reshape(seq_len*batch_size, self.causes_dim)
        )
        self.w_c -= lr_c * delta_w_c
        
        delta_w_r = torch.mm(
            self.error_h[1:].reshape((seq_len-1)*batch_size, self.states_dim).T,
            torch.tanh(self.h_post[:-1].reshape((seq_len-1)*batch_size, self.states_dim))
        )
        self.w_r -= lr_r * delta_w_r
            
        delta_b_r = torch.sum(self.error_h.reshape(seq_len*batch_size, self.states_dim), axis=0)
        self.b_r -= lr_r * delta_b_r
    
####################################################################################################################################

class PC_RNN_HC_M(nn.Module):
    
    def __init__(self, causes_dim, states_dim, output_dim, factor_dim, tau_h, alpha_x, alpha_h):
        super(PC_RNN_HC_M, self).__init__()
        
        self.causes_dim = causes_dim
        self.output_dim = output_dim
        self.states_dim = states_dim
        self.factor_dim = factor_dim
        self.tau_h = tau_h  # Time constant of the network
        self.alpha_x = alpha_x  # Update rate coefficient of the hidden state layer
        self.alpha_h = alpha_h  # Update rate coefficient of the hidden causes (input) layer
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim) / 10
        
        # Recurrent weights initialization
        self.w_pd = torch.randn(self.states_dim, self.factor_dim) / np.sqrt(self.factor_dim)
        self.w_fd = torch.randn(self.states_dim, self.factor_dim) / np.sqrt(self.factor_dim)
        self.w_cd = torch.nn.Softmax(1)(0.5*torch.randn(self.causes_dim, self.factor_dim))*self.factor_dim
        self.b_r = torch.randn(self.states_dim) / 10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        self.c = None
        
    def forward(self, x, c_init, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - c_init : initial hidden causes (input), Tensor of shape (batch_size, states_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store as attributes the different activations, should be set to
            True if learning is going to be performed on the target sequence, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len+1, batch_size, self.states_dim)
            cs = torch.zeros(seq_len+1, batch_size, self.causes_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        c = c_init
        
        if store:
            cs[0] = c
            h_posts[0] = old_h_post
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h_prior = (1-1/self.tau_h) * old_h_post + (1/self.tau_h) * (
                torch.mm(
                    torch.mm(
                        torch.tanh(old_h_post),
                        self.w_pd
                    ) * torch.mm(
                        c,
                        self.w_cd
                    ),
                    self.w_fd.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                h_priors[t] = h_prior.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
            
            # Compute the error on the sensory level
            error = x_pred - x[t]
            errors[t] = error

            # Bottom-up pass
            if self.alpha_x>0:
            
                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*(1-torch.tanh(h_prior)**2)*torch.mm(error, self.w_o)
                if store:
                    h_posts[t+1] = h_post.detach()

                # Compute the error on the hidden state level
                error_h = h_prior - h_post
                if store:
                    error_hs[t] = error_h.detach()

                # Infer c according to its past value and the error on the hidden state level
                c = c - self.alpha_h*torch.mm(
                    torch.mm(
                        torch.tanh(old_h_post),
                        self.w_pd
                    )* torch.mm(
                        error_h,
                        self.w_fd
                    ),
                    self.w_cd.T
                )
                if store:
                    cs[t+1] = c

                old_h_post = h_post
                
            else:
                old_h_post = h_prior
        
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.error_h = error_hs
            self.h_prior = h_priors
            self.h_post = h_posts
            self.c = cs
                             
        return errors
    
    def learn(self, lr, lr_r, lr_c):
        """
        The implementation provided here uses the sequence length as batch dimension to speed up computations
          Still the learning rule does not use any non-local or past/future information and thus can be applied
          in parallel with the forward computations of the RNN
        Parameters:
        - lr: learning rate of the output_weights, double
        - lr_r: learning rate of the recurrent weights, double
        - lr_c: learning rate of the input weights, double
        """
        
        seq_len, batch_size, _ = self.x_pred.shape
        nbatch = seq_len*batch_size

        # Output weights
        delta_o = lr * torch.mm(
            self.error.reshape(nbatch, self.output_dim).T, 
            torch.tanh(self.h_prior.reshape(nbatch, self.states_dim))
        )
        
        self.w_o -= delta_o
                    
        delta_b_o = torch.sum(self.error.reshape(seq_len*batch_size, self.output_dim), axis=0)
        self.b_o -= lr * delta_b_o
        
        # Recurrent weights
        delta_pd = lr_r * torch.mm(
            (
                torch.mm(
                    self.error_h.reshape(nbatch, self.states_dim),
                    self.w_fd
                ) * \
                torch.mm(
                    self.c[:-1].reshape(nbatch, self.causes_dim),
                    self.w_cd
                )
            ).reshape(nbatch, self.factor_dim).T,
            torch.tanh(self.h_post[:-1]).reshape(nbatch, self.states_dim)
        )
        
        self.w_pd -= delta_pd.T
        
        delta_cd = lr_c * torch.mm(
            (
                torch.mm(
                    self.error_h.reshape(nbatch, self.states_dim),
                    self.w_fd
                ) * \
                torch.mm(
                    torch.tanh(self.h_post[:-1]).reshape(nbatch, self.states_dim),
                    self.w_pd
                )
            ).reshape(nbatch, self.factor_dim).T,
            self.c[:-1].reshape(nbatch, self.causes_dim)
        )
        self.w_cd -= delta_cd.T
        
        delta_fd = lr_r * torch.mm(
            (
                torch.mm(
                    torch.tanh(self.h_post[:-1]).reshape(nbatch, self.states_dim),
                    self.w_pd
                ) * \
                torch.mm(
                    self.c[:-1].reshape(nbatch, self.causes_dim),
                    self.w_cd
                )
            ).reshape(nbatch, self.factor_dim).T,
            torch.tanh(self.error_h).reshape(nbatch, self.states_dim)
        )
        
        self.w_fd -= delta_fd.T
        
        delta_b_r = torch.sum(self.error_h.reshape(seq_len*batch_size, self.states_dim), axis=0)
        self.b_r -= lr_r * delta_b_r
        
####################################################################################################################################

class PC_RNN_HC_A_RS(nn.Module):
    
    def __init__(self, causes_dim, states_dim, output_dim, tau_h, alpha_x, alpha_h):
        super(PC_RNN_HC_A_RS, self).__init__()
        
        self.causes_dim = causes_dim
        self.output_dim = output_dim
        self.states_dim = states_dim
        self.tau_h = tau_h  # Time constant of the network
        self.alpha_x = alpha_x  # Update rate coefficient of the hidden state layer
        self.alpha_h = alpha_h  # Update rate coefficient of the hidden causes (input) layer
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim) / 10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_r = torch.randn(self.states_dim) / 10

        # Input weights initialization
        self.w_c = torch.randn(self.states_dim, self.causes_dim) / self.causes_dim
        
        # Predictions, states and errors are temporarily stored for batch learning
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        self.c = None

        self.k=0
        self.reset_learning()
        
    def forward(self, x, c_init, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - c_init : initial hidden causes (input), Tensor of shape (batch_size, states_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store as attributes the different activations, should be set to
            True if learning is going to be performed on the target sequence, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            cs = torch.zeros(seq_len, batch_size, self.causes_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        c = c_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h_prior = (1-1/self.tau_h) * old_h_post + (1/self.tau_h) * (
                torch.mm(
                    torch.tanh(old_h_post),
                    self.w_r.T
                ) + \
                torch.mm(
                    c,
                    self.w_c.T
                ) + \
                self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                h_priors[t] = h_prior.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
            
            # Compute the error on the output level
            error = x_pred - x[t]
            errors[t] = error
            
            # Bottom-up pass
            if self.alpha_x > 0:

                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*(1-torch.tanh(h_prior)**2)*torch.mm(error, self.w_o)
                if store:
                    h_posts[t] = h_post.detach()

                # Compute the error on the hidden state level
                error_h = h_prior - h_post
                if store:
                    error_hs[t] = error_h.detach()

                # Infer c according to its past value and the error on the hidden state level
                c = c - self.alpha_h*torch.mm(error_h, self.w_c)
                if store:
                    cs[t] = c
                             
                old_h_post = h_post
            
            else:
                old_h_post = h_prior
                
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.error_h = error_hs
            self.h_prior = h_priors
            self.h_post = h_posts
            self.c = cs
                             
        return errors

    def switch_task(self):
        self.k += 1
        self.reset_learning()
    
    def reset_learning(self):
        self.past_error=float('inf')
        self.noise = torch.zeros(self.states_dim)
    
    def learn(self, lr, lr_r, sigma_r):
        """
        The implementation provided here uses the sequence length as batch dimension to speed up computations
          Still the learning rule does not use any non-local or past/future information and thus can be applied
          in parallel with the forward computations of the RNN
        Parameters:
        - lr: learning rate of the output_weights, double
        - lr_r: learning rate of the recurrent weights, double
        - sigma_r: standard deviation of the random search, double
        """
                
        seq_len, batch_size, _ = self.x_pred.shape
        
        delta_w_o = torch.mm(
            self.error.reshape(seq_len*batch_size, self.output_dim).T, 
            torch.tanh(self.h_prior.reshape(seq_len*batch_size, self.states_dim))
        )
        self.w_o -= lr * delta_w_o

        delta_b_o = torch.sum(self.error.reshape(seq_len*batch_size, self.output_dim), axis=0)
        self.b_o -= lr * delta_b_o
               
        delta_w_r = torch.mm(
            self.error_h[1:].reshape((seq_len-1)*batch_size, self.states_dim).T,
            torch.tanh(self.h_post[:-1].reshape((seq_len-1)*batch_size, self.states_dim))
        )
        self.w_r -= lr_r * delta_w_r
            
        delta_b_r = torch.sum(self.error_h.reshape(seq_len*batch_size, self.states_dim), axis=0)
        self.b_r -= lr_r * delta_b_r
        
        # Whether to save the last applied noise
        new_error = torch.mean(torch.sqrt(torch.sum(self.error**2, axis=2)))
        if self.past_error > new_error:
            self.past_error = new_error
        else:
            self.w_c[:, self.k] -= self.noise
            
        # Apply new noise on the output weights
        self.noise = torch.rand(self.states_dim) * sigma_r
        self.w_c[:, self.k] += self.noise
    
####################################################################################################################################

class PC_RNN_HC_M_RS(nn.Module):
    
    def __init__(self, causes_dim, states_dim, output_dim, factor_dim, tau_h, alpha_x, alpha_h):
        super(PC_RNN_HC_M_RS, self).__init__()
        
        self.causes_dim = causes_dim
        self.output_dim = output_dim
        self.states_dim = states_dim
        self.factor_dim = factor_dim
        self.tau_h = tau_h  # Time constant of the network
        self.alpha_x = alpha_x  # Update rate coefficient of the hidden state layer
        self.alpha_h = alpha_h  # Update rate coefficient of the hidden causes (input) layer
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim) / 10
        
        # Recurrent weights initialization
        self.w_pd = torch.randn(self.states_dim, self.factor_dim) / np.sqrt(self.factor_dim)
        self.w_fd = torch.randn(self.states_dim, self.factor_dim) / np.sqrt(self.factor_dim)
        self.w_cd = torch.nn.Softmax(1)(0.5*torch.randn(self.causes_dim, self.factor_dim))*self.factor_dim
        self.b_r = torch.randn(self.states_dim) / 10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        self.c = None
        
        self.k=0
        self.reset_learning()
        
    def forward(self, x, c_init, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - c_init : initial hidden causes (input), Tensor of shape (batch_size, states_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store as attributes the different activations, should be set to
            True if learning is going to be performed on the target sequence, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len+1, batch_size, self.states_dim)
            cs = torch.zeros(seq_len+1, batch_size, self.causes_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        c = c_init
        
        if store:
            cs[0] = c
            h_posts[0] = old_h_post
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h_prior = (1-1/self.tau_h) * old_h_post + (1/self.tau_h) * (
                torch.mm(
                    torch.mm(
                        torch.tanh(old_h_post),
                        self.w_pd
                    ) * torch.mm(
                        c,
                        self.w_cd
                    ),
                    self.w_fd.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                h_priors[t] = h_prior.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
            
            # Compute the error on the sensory level
            error = x_pred - x[t]
            errors[t] = error

            # Bottom-up pass
            if self.alpha_x>0:
            
                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*(1-torch.tanh(h_prior)**2)*torch.mm(error, self.w_o)
                if store:
                    h_posts[t+1] = h_post.detach()

                # Compute the error on the hidden state level
                error_h = h_prior - h_post
                if store:
                    error_hs[t] = error_h.detach()

                # Infer c according to its past value and the error on the hidden state level
                c = c - self.alpha_h*torch.mm(
                    torch.mm(
                        torch.tanh(old_h_post),
                        self.w_pd
                    )* torch.mm(
                        error_h,
                        self.w_fd
                    ),
                    self.w_cd.T
                )
                if store:
                    cs[t+1] = c

                old_h_post = h_post
                
            else:
                old_h_post = h_prior
        
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.error_h = error_hs
            self.h_prior = h_priors
            self.h_post = h_posts
            self.c = cs
                             
        return errors

    def switch_task(self):
        self.k += 1
        self.reset_learning()
        
    def reset_learning(self):
        self.past_error=float('inf')
        self.noise = torch.zeros(self.factor_dim)
        
    def learn(self, lr, lr_r, sigma_r):
        """
        The implementation provided here uses the sequence length as batch dimension to speed up computations
          Still the learning rule does not use any non-local or past/future information and thus can be applied
          in parallel with the forward computations of the RNN
        Parameters:
        - lr: learning rate of the output_weights, double
        - lr_r: learning rate of the recurrent weights, double
        - sigma_r: standard deviation of the random search, double
        """
        
        seq_len, batch_size, _ = self.x_pred.shape
        nbatch = seq_len*batch_size

        # Output weights
        delta_o = lr * torch.mm(
            self.error.reshape(nbatch, self.output_dim).T, 
            torch.tanh(self.h_prior.reshape(nbatch, self.states_dim))
        )
        
        self.w_o -= delta_o
                    
        delta_b_o = torch.sum(self.error.reshape(seq_len*batch_size, self.output_dim), axis=0)
        self.b_o -= lr * delta_b_o
        
        # Recurrent weights
        delta_pd = lr_r * torch.mm(
            (
                torch.mm(
                    self.error_h.reshape(nbatch, self.states_dim),
                    self.w_fd
                ) * \
                torch.mm(
                    self.c[:-1].reshape(nbatch, self.causes_dim),
                    self.w_cd
                )
            ).reshape(nbatch, self.factor_dim).T,
            torch.tanh(self.h_post[:-1]).reshape(nbatch, self.states_dim)
        )
        
        self.w_pd -= delta_pd.T
        
        delta_fd = lr_r * torch.mm(
            (
                torch.mm(
                    torch.tanh(self.h_post[:-1]).reshape(nbatch, self.states_dim),
                    self.w_pd
                ) * \
                torch.mm(
                    self.c[:-1].reshape(nbatch, self.causes_dim),
                    self.w_cd
                )
            ).reshape(nbatch, self.factor_dim).T,
            torch.tanh(self.error_h).reshape(nbatch, self.states_dim)
        )
        
        self.w_fd -= delta_fd.T
        
        delta_b_r = torch.sum(self.error_h.reshape(seq_len*batch_size, self.states_dim), axis=0)
        self.b_r -= lr_r * delta_b_r
        
        # Whether to save the last applied noise
        new_error = torch.mean(torch.sqrt(torch.sum(self.error**2, axis=2)))
        if self.past_error > new_error:
            self.past_error = new_error
        else:
            self.w_cd[self.k] -= self.noise
            
        # Apply new noise on the output weights
        self.noise = torch.rand(self.factor_dim) * sigma_r
        self.w_cd[self.k] += self.noise