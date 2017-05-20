import numpy as np
from numpy.testing import assert_allclose


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    c = np.max(x, axis=-1, keepdims=True)
    exps = np.exp(x - c)
    sum_exps = np.sum(exps, axis=-1, keepdims=True)
    x = exps / sum_exps

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


TEST_CASES = [
  {
    "input": np.array([0, 0]),
    "expected": np.array([0.5, 0.5])
  },
  {
    "input": np.array([1, 1]),
    "expected": np.array([0.5, 0.5])
  },
  {
    "input": np.array([1.0, 1e-3]),
    "expected": np.array([0.7308619, 0.26913807])
  },
  {
    "input": np.array([3.0, 1.0, 0.2]),
    "expected": np.array([ 0.8360188, 0.11314284, 0.05083836])
  },
  {
    "input": np.array(
      [[1, 2, 3, 6],
       [2, 4, 5, 6],
       [3, 8, 7, 6]]),
    "expected": np.array(
      [[ 0.006269,  0.01704 ,  0.04632 ,  0.93037 ],
       [ 0.012038,  0.088947,  0.241783,  0.657233],
       [ 0.004462,  0.662272,  0.243636,  0.089629]])
  },
  {
    "input": np.array(
      [[ 0.31323624,  0.7810351 ,  0.26183059,  0.09174578,  0.09806706],
       [ 0.28981829,  0.03154328,  0.99442807,  0.4591928 ,  0.42556593],
       [ 0.06799825,  0.89438807,  0.68276332,  0.89185543,  0.37638809],
       [ 0.49131144,  0.03873597,  0.91306311,  0.2533448 ,  0.24115072],
       [ 0.38297911,  0.23184308,  0.88202174,  0.42546236,  0.78325552]]),
    "expected": np.array(
      [[ 0.19402037,  0.30974891,  0.18429864,  0.15547309,  0.15645899],
       [ 0.16325474,  0.12609515,  0.33027366,  0.19338564,  0.18699081],
       [ 0.11396294,  0.26041151,  0.21074279,  0.25975282,  0.15512995],
       [ 0.21152728,  0.13452883,  0.32250081,  0.16673194,  0.16471114],
       [ 0.16549414,  0.1422804 ,  0.27259261,  0.17267635,  0.2469565 ]])
  }
]

def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    for test_case in TEST_CASES:
      test_input = test_case["input"]
      test_output = softmax(test_input)
      expected_output = test_case["expected"]
      assert_allclose(
        test_output, expected_output,
        rtol=1e-07, atol=1e-06
      )
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
