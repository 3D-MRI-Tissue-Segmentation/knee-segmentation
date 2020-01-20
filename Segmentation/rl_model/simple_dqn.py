import os
import tensorflow as tf
import numpy as np

class DQN:

    def __init__(self, lr, n_actions, name, input_dims,
                 fc_dims=[256, 256], checkpoint_dir="checkpoints/dqn"):
        self.lr = lr
        self.name = name

        self.input_dims = input_dims
        self.fc_dims = fc_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(checkpoint_dir, "dqn.ckpt")

        self.model = self.build_network()

    def build_network(self):
        
        input = tf.keras.Input()



        return model

    def save(self):
        pass

    def load(self):
        pass
        