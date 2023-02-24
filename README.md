# PongML - DQL Branch
This is the most up-to-date branch currently in development. I am working on finishing the implementation of a Deep Q-Learning Network that uses Keras. The network architecture is primarily layers of convolution with 1 max pooling layer and a fully connected layer before the output layer that predicts the best action.
This version features a file management script that automatically runs to keep the model data - including counts for episodes and frames, and the value of epsilon - up to date between training sessions.
I am still tweaking the reward system and other hyperparameters periodically.
Next update will include an optimized training timing to enhance the watchability of the simulation. Of relevance will also be the impact that this change has on model performance.
