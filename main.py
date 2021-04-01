import numpy as np

#Creating a class for our neural net

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2*np.random.random((3,1)) - 1

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x*(1-x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for i in range(training_iterations):

            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T , error*self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self , inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

if __name__ == "__main__":

    #Initialise Neural Network

    neural_network = NeuralNetwork()

    print("Random Synaptic Weights:")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0,0,1],
                                  [1,1,1],
                                  [1,0,1],
                                  [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    neural_network.train(training_inputs, training_outputs, 10000)

    print("Synaptic weights after training:")
    print(neural_network.synaptic_weights)

    A = str(input("First Input:"))
    B = str(input("Second Input:"))
    C = str(input("Third Input:"))

    print("New Scenario: Input Data=", A,B,C)
    print("Output Data:")
    print(neural_network.think(np.array([A , B , C])))
