from MiniFlow.miniflow import *

__author__ = 'Ibis'

# """
# This script builds and runs a graph with miniflow.
#
# There is no need to change anything to solve this quiz!
#
# However, feel free to play with the network! Can you also
# build a network that solves the equation below?
#
# (x + y) + y
# """
#
# from miniflow import *
#
# #----------------Script 1----------------#
#
# x, y = Input(), Input()
#
# f = Add(x, y)
#
# feed_dict = {x: 10, y: 5}
#
# sorted_neurons = topological_sort(feed_dict)
# output = forward_pass(f, sorted_neurons)
#
# # NOTE: because topological_sort set the values for the `Input` neurons we could also access
# # the value for x with x.value (same goes for y).
# print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))
#
# #----------------Script 2----------------#
#
# x, y, z = Input(), Input(), Input()
#
# f = Add(x, y, z)
#
# feed_dict = {x: 4, y: 5, z: 10}
#
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)
#
# # should output 19
# print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
#
# #----------------Script 3----------------#
#
# x, y, z = Input(), Input(), Input()
#
# f = Mul(x, y, z)
#
# feed_dict = {x: 4, y: 5, z: 10}
#
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)
#
# # should output 19
# print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
#
# #----------------Script 4----------------#
# x, y, z = Input(), Input(), Input()
# inputs = [x, y, z]
#
# weight_x, weight_y, weight_z = Input(), Input(), Input()
# weights = [weight_x, weight_y, weight_z]
#
# bias = Input()
#
# f = Linear(inputs, weights, bias)
#
# feed_dict = {
# 	x: 6,
# 	y: 14,
# 	z: 3,
# 	weight_x: 0.5,
# 	weight_y: 0.25,
# 	weight_z: 1.4,
# 	bias: 2
# }
#
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)
#
# print(output) # should be 12.7 with this example

#----------------Script 5----------------#

"""
This scripts demonstrates how the new MiniFlow works!

Update the Linear class in miniflow.py to work with
numpy vectors (arrays) and matrices.

Test your code here!
"""

import numpy as np
from miniflow import *

inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)

x = np.array([[-1., -2.], [-1, -2]])
w = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])

feed_dict = {inputs: x, weights: w, bias: b}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

"""
Output should be:
[[-9., 4.],
[-9., 4.]]
"""
print(output)

#----------------Script 5----------------#

"""
This network feeds the output of a linear transform
to the sigmoid function.

Finish implementing the Sigmoid class in miniflow.py!

Feel free to play around with this network, too!
"""

import numpy as np
from miniflow import *

inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)
g = Sigmoid(f)

x = np.array([[-1., -2.], [-1, -2]])
w = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])

feed_dict = {inputs: x, weights: w, bias: b}

graph = topological_sort(feed_dict)
output = forward_pass(g, graph)

"""
Output should be:
[[  1.23394576e-04   9.82013790e-01]
 [  1.23394576e-04   9.82013790e-01]]
"""
print(output)