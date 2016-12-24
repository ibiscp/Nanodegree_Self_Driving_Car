#--------------------------------------------Exercise Neuron-----------------------------------------------------#

# """
# Bonus Challenge!
#
# Write your code in Add (scroll down).
# """
#
# class Neuron:
#     def __init__(self, inbound_neurons=[], label=''):
#         # An optional description of the neuron - most useful for outputs.
#         self.label = label
#         # Neurons from which this Node receives values
#         self.inbound_neurons = inbound_neurons
#         # Neurons to which this Node passes values
#         self.outbound_neurons = []
#         # A calculated value
#         self.value = None
#         # Add this node as an outbound node on its inputs.
#         for n in self.inbound_neurons:
#             n.outbound_neurons.append(self)
#
#
#     # These will be implemented in a subclass.
#     def forward(self):
#         """
#         Forward propagation.
#
#         Compute the output value based on `inbound_neurons` and
#         store the result in self.value.
#         """
#         raise NotImplemented
#
#     def backward(self):
#         """
#         Backward propagation.
#
#         Compute the gradient of the current Neuron with respect
#         to the input neurons. The gradient of the loss with respect
#         to the current Neuron should already be computed in the `gradients`
#         attribute of the output neurons.
#         """
#         raise NotImplemented
#
# class Input(Neuron):
#     def __init__(self):
#         # An Input Neuron has no inbound neurons,
#         # so no need to pass anything to the Neuron instantiator
#         Neuron.__init__(self)
#
#     # NOTE: Input Neuron is the only Neuron where the value
#     # may be passed as an argument to forward().
#     #
#     # All other Neuron implementations should get the value
#     # of the previous neurons from self.inbound_neurons
#     #
#     # Example:
#     # val0 = self.inbound_neurons[0].value
#     def forward(self, value=None):
#         # Overwrite the value if one is passed in.
#         if value is not None:
#             self.value = value
#
#     def backward(self):
#         # An Input Neuron has no inputs so we refer to ourself
#         # for the gradient
#         self.gradients = {self: 0}
#         for n in self.outbound_neurons:
#             self.gradients[self] += n.gradients[self]
#
#
# """
# Can you augment the Add class so that it accepts
# any number of neurons as input?
#
# Hint: this may be useful:
# https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists
# """
# class Add(Neuron):
#     # You may need to change this...
#     def __init__(self, *inputs):
#         Neuron.__init__(self, inputs)
#
#     def forward(self):
#         """
#         For reference, here's the old way from the last
#         quiz. You'll want to write code here.
#         """
#         # x_value = self.inbound_neurons[0].value
#         # y_value = self.inbound_neurons[1].value
#         # self.value = x_value + y_value
#         summ = 0
#         for i in self.inbound_neurons:
#             summ = summ + i.value
#
#         self.value = summ
#
# class Mul(Neuron):
#     # You may need to change this...
#     def __init__(self, *inputs):
#         Neuron.__init__(self, inputs)
#
#     def forward(self):
#         """
#         For reference, here's the old way from the last
#         quiz. You'll want to write code here.
#         """
#         # x_value = self.inbound_neurons[0].value
#         # y_value = self.inbound_neurons[1].value
#         # self.value = x_value + y_value
#         mul = 1
#         for i in self.inbound_neurons:
#             mul = mul * i.value
#
#         self.value = mul
#
# class Linear(Neuron):
#     def __init__(self, inputs, weights, bias):
#         Neuron.__init__(self, inputs)
#
#         # NOTE: The weights and bias properties here are not
#         # numbers, but rather references to other neurons.
#         # The weight and bias values are stored within the
#         # respective neurons.
#         self.weights = weights
#         self.bias = bias
#
#     def forward(self):
#         """
#         Set self.value to the value of the linear function output.
#
#         Your code goes here!
#         """
#         self.value = self.bias.value
#         for i in range(len(self.inbound_neurons)):
#             input_x = self.inbound_neurons[i].value
#             weight_w = self.weights[i].value
#             self.value = self.value + input_x * weight_w
#
# def topological_sort(feed_dict):
#     """
#     Sort the neurons in topological order using Kahn's Algorithm.
#
#     `feed_dict`: A dictionary where the key is a `Input` Neuron and the value is the respective value feed to that Neuron.
#
#     Returns a list of sorted neurons.
#     """
#
#     input_neurons = [n for n in feed_dict.keys()]
#
#     G = {}
#     neurons = [n for n in input_neurons]
#     while len(neurons) > 0:
#         n = neurons.pop(0)
#         if n not in G:
#             G[n] = {'in': set(), 'out': set()}
#         for m in n.outbound_neurons:
#             if m not in G:
#                 G[m] = {'in': set(), 'out': set()}
#             G[n]['out'].add(m)
#             G[m]['in'].add(n)
#             neurons.append(m)
#
#     L = []
#     S = set(input_neurons)
#     while len(S) > 0:
#         n = S.pop()
#
#         if isinstance(n, Input):
#             n.value = feed_dict[n]
#
#         L.append(n)
#         for m in n.outbound_neurons:
#             G[n]['out'].remove(m)
#             G[m]['in'].remove(n)
#             # if no other incoming edges add to S
#             if len(G[m]['in']) == 0:
#                 S.add(m)
#     return L
#
#
# def forward_pass(output_Neuron, sorted_neurons):
#     """
#     Performs a forward pass through a list of sorted neurons.
#
#     Arguments:
#
#         `output_Neuron`: A Neuron in the graph, should be the output Neuron (have no outgoing edges).
#         `sorted_neurons`: a topologically sorted list of neurons.
#
#     Returns the output Neuron's value
#     """
#
#     for n in sorted_neurons:
#         n.forward()
#
#     return output_Neuron.value

#--------------------------------------------Exercise Layer-----------------------------------------------------#

# """
# Modify Linear#forward so that it linearly transforms
# input matrices, weights matrices and a bias vector to
# an output.
# """
#
# import numpy as np
#
# class Layer:
#     def __init__(self, inbound_layers=[]):
#         self.inbound_layers = inbound_layers
#         self.value = None
#         self.outbound_layers = []
#         for layer in inbound_layers:
#             layer.outbound_layers.append(self)
#
#     def forward(self):
#         raise NotImplementedError
#
#
# class Input(Layer):
#     """
#     While it may be strange to consider an input a layer when
#     an input is only an individual node in a layer, for the sake
#     of simpler code we'll still use Layer as the base class.
#
#     Think of Input as collating many individual input nodes into
#     a Layer.
#     """
#     def __init__(self):
#         # An Input layer has no inbound layers,
#         # so no need to pass anything to the Layer instantiator
#         Layer.__init__(self)
#
#     def forward(self):
#         # Do nothing because nothing is calculated.
#         pass
#
#
# class Linear(Layer):
#     def __init__(self, inbound_layer, weights, bias):
#         # Notice the ordering of the input layers passed to the
#         # Layer constructor.
#         Layer.__init__(self, [inbound_layer, weights, bias])
#
#     def forward(self):
#         """
#         Set the value of this layer to the linear transform output.
#
#         Your code goes here!
#         """
#         inputs = self.inbound_layers[0].value
#         weights = self.inbound_layers[1].value
#         bias = self.inbound_layers[2].value
#         self.value = np.dot(inputs, weights) + bias
#
#
# def topological_sort(feed_dict):
#     """
#     Sort the layers in topological order using Kahn's Algorithm.
#
#     `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.
#
#     Returns a list of sorted layers.
#     """
#
#     input_layers = [n for n in feed_dict.keys()]
#
#     G = {}
#     layers = [n for n in input_layers]
#     while len(layers) > 0:
#         n = layers.pop(0)
#         if n not in G:
#             G[n] = {'in': set(), 'out': set()}
#         for m in n.outbound_layers:
#             if m not in G:
#                 G[m] = {'in': set(), 'out': set()}
#             G[n]['out'].add(m)
#             G[m]['in'].add(n)
#             layers.append(m)
#
#     L = []
#     S = set(input_layers)
#     while len(S) > 0:
#         n = S.pop()
#
#         if isinstance(n, Input):
#             n.value = feed_dict[n]
#
#         L.append(n)
#         for m in n.outbound_layers:
#             G[n]['out'].remove(m)
#             G[m]['in'].remove(n)
#             # if no other incoming edges add to S
#             if len(G[m]['in']) == 0:
#                 S.add(m)
#     return L
#
#
# def forward_pass(output_layer, sorted_layers):
#     """
#     Performs a forward pass through a list of sorted Layers.
#
#     Arguments:
#
#         `output_layer`: A Layer in the graph, should be the output layer (have no outgoing edges).
#         `sorted_layers`: a topologically sorted list of layers.
#
#     Returns the output layer's value
#     """
#
#     for n in sorted_layers:
#         n.forward()
#
#     return output_layer.value

#--------------------------------------------Exercise Sigmoid-----------------------------------------------------#

"""
Fix the Sigmoid class so that it computes the sigmoid function
on the forward pass!

Scroll down to get started.
"""

import numpy as np

class Layer:
    def __init__(self, inbound_layers=[]):
        self.inbound_layers = inbound_layers
        self.value = None
        self.outbound_layers = []
        for layer in inbound_layers:
            layer.outbound_layers.append(self)

    def forward():
        raise NotImplementedError

    def backward():
        raise NotImplementedError


class Input(Layer):
    def __init__(self):
        # An Input layer has no inbound layers,
        # so no need to pass anything to the Layer instantiator
        Layer.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input Layer has no inputs so we refer to ourself
        # for the gradient
        self.gradients = {self: 0}
        for n in self.outbound_Layers:
            self.gradients[self] += n.gradients[self]


class Linear(Layer):
    def __init__(self, inbound_layer, weights, bias):
        # Notice the ordering of the input layers passed to the
        # Layer constructor.
        Layer.__init__(self, [inbound_layer, weights, bias])

    def forward(self):
        inputs = self.inbound_layers[0].value
        weights = self.inbound_layers[1].value
        bias = self.inbound_layers[2].value
        self.value = np.dot(inputs, weights) + bias


class Sigmoid(Layer):
    """
    You need to fix the `_sigmoid` and `forward` methods.
    """
    def __init__(self, layer):
        Layer.__init__(self, [layer])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.

        Return the result of the sigmoid function.

        Your code here!
        """
        return 1. / (1. + np.exp(-x))


    def forward(self):
        """
        Set the value of this layer to the result of the
        sigmoid function, `_sigmoid`.

        Your code here!
        """
        # This is a dummy value to prevent numpy errors
        # if you test without changing this method.
        input_value = self.inbound_layers[0].value
        self.value = self._sigmoid(input_value)


def topological_sort(feed_dict):
    """
    Sort the layers in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.

    Returns a list of sorted layers.
    """

    input_layers = [n for n in feed_dict.keys()]

    G = {}
    layers = [n for n in input_layers]
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_layers:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)

    L = []
    S = set(input_layers)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_layers:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_layer, sorted_layers):
    """
    Performs a forward pass through a list of sorted Layers.

    Arguments:

        `output_layer`: A Layer in the graph, should be the output layer (have no outgoing edges).
        `sorted_layers`: a topologically sorted list of layers.

    Returns the output layer's value
    """

    for n in sorted_layers:
        n.forward()

    return output_layer.value