import numpy as np


class NeuralNetwork:
    '''Shallow Neural Network for likelihood estimation '''
    def __init__(self, number_of_inputs: int, size_of_hidden_layer:int, number_of_outputs: int, epochs: int):
        """
            Initializes instances of NeuralNetwork

        Args:
            number_of_inputs (int)
            size_of_hidden_layer (int)
            number_of_outputs (int)
            epochs (int): How many epochs will stochastic gradient descent run
        """

        self.number_of_inputs = number_of_inputs
        self.size_of_hidden_layer = size_of_hidden_layer
        self.number_of_outputs = number_of_outputs
        self.epochs = epochs

        np.random.seed(1)

        std_dev_hl = np.sqrt(1 / (self.size_of_hidden_layer + self.number_of_inputs))
        std_dev_ol = np.sqrt(1 / (self.size_of_hidden_layer + self.number_of_outputs))
        initial_weights_hl = np.random.normal(loc = 0,  #* Initial Weights Hidden Layer
                                                scale = std_dev_hl,
                                                size = (self.size_of_hidden_layer,
                                                        self.number_of_inputs))
        initial_weights_ol = np.random.normal(loc = 0, #* Initial Weights Output Layer
                                                scale = std_dev_ol,
                                                size = (self.number_of_outputs,
                                                        self.size_of_hidden_layer))

        self.hl_weights = initial_weights_hl.copy() #* Hidden Layer Weights Cross Entropy
        self.hl_biases = np.zeros((self.size_of_hidden_layer, 1)) #* HL Biases Cross Entropy

        self.ol_weights = initial_weights_ol.copy() #* Output Layer Cross Entropy
        self.ol_bias = 0

        self.cost_epoch_array = []

    def gradient(self, x):
        """ Returns the gradient of the neural network with respect to its parameters.

        Args:
            x : input vector of data.

        Returns:
            An array with the derivatives of the neural network with respect to its parameters.
        """
        w1 = self.hl_weights @ x + self.hl_biases
        z1 = np.maximum(0, w1) #? RELU ACTIVATION
        w2 = self.ol_weights @ z1 + self.ol_bias

        y = 1 / (1 + np.exp(-w2)) #? SIGMOID ACTIVATION

        v2 = y * (1 - y) #? Sigmoid Rerivative Based On Output y

        u1 = self.ol_weights.T @ v2

        v1 = u1 * np.where(w1 > 0, 1, 0)

        d_A2 = v2 @ z1.T
        d_B2 = v2

        d_A1 = v1 @ x.T
        d_B1 = v1

        fin = np.array([d_A1, d_B1, d_A2, d_B2], dtype =object), y.item()
        return fin
    