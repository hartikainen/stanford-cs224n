#!/usr/bin/env python

import numpy as np
from numpy.testing import assert_allclose


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1.0 / (1.0 + np.exp(-x))
    ### END YOUR CODE

    return s


def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ### YOUR CODE HERE
    ds = (1 - s) * s
    ### END YOUR CODE

    return ds


def test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print "You should verify these results by hand!\n"


TEST_CASES = [
  {
    "fn": sigmoid,
    "input": np.array([[0, 0], [1,1]]),
    "expected": np.array(
      [[ 0.5     ,  0.5     ],
       [ 0.731059,  0.731059]])
  },
  {
    "fn": sigmoid_grad,
    "input": sigmoid(np.array([[0, 0], [1,1]])),
    "expected": np.array(
      [[ 0.25      ,  0.25      ],
       [ 0.19661193,  0.19661193]])
  },
  {
    "fn": sigmoid,
    "input": np.array(range(12)).reshape(3, 4),
    "expected": np.array(
      [[ 0.5     ,  0.731059,  0.880797,  0.952574],
       [ 0.982014,  0.993307,  0.997527,  0.999089],
       [ 0.999665,  0.999877,  0.999955,  0.999983]])
  },
  {
    "fn": sigmoid_grad,
    "input": sigmoid(np.array(range(12)).reshape(3, 4)),
    "expected": np.array(
      [[  2.50000000e-01,   1.96611933e-01,   1.04993585e-01,
          4.51766597e-02],
       [  1.76627062e-02,   6.64805667e-03,   2.46650929e-03,
          9.10221180e-04],
       [  3.35237671e-04,   1.23379350e-04,   4.53958077e-05,
          1.67011429e-05]])
  },
  {
    "fn": sigmoid,
    "input": np.array(range(0, -12, -1)).reshape(3, 4),
    "expected": np.array(
      [[  5.000000e-01,   2.689414e-01,   1.192029e-01,   4.742587e-02],
       [  1.798621e-02,   6.692851e-03,   2.472623e-03,   9.110512e-04],
       [  3.353501e-04,   1.233946e-04,   4.539787e-05,   1.670142e-05]])
  },
  {
    "fn": sigmoid_grad,
    "input": sigmoid(np.array(range(0, -12, -1)).reshape(3, 4)),
    "expected": np.array(
      [[  2.50000000e-01,   1.96611933e-01,   1.04993585e-01,
          4.51766597e-02],
       [  1.76627062e-02,   6.64805667e-03,   2.46650929e-03,
          9.10221180e-04],
       [  3.35237671e-04,   1.23379350e-04,   4.53958077e-05,
          1.67011429e-05]])
  }
]

def test_sigmoid():
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    for i, test_case in enumerate(TEST_CASES):
      test_input = test_case["input"]
      test_fn = test_case["fn"]
      test_output = test_fn(test_input)
      expected_output = test_case["expected"]
      assert_allclose(
        test_output, expected_output,
        rtol=1e-07, atol=1e-06
      )
      print("Test {} passed".format(i))
    ### END YOUR CODE


if __name__ == "__main__":
    test_sigmoid_basic();
    test_sigmoid()
