import tensorflow as tf


if __name__ == '__main__':
    # print(tf.__version__)

    # Create tensor
    scalar = tf.constant(7)
    print(scalar)
    # Check the number of dimension
    print(scalar.ndim)
    
    # Create a vector
    vector = tf.constant([10, 10])
    print(vector)
    # Check the number of dimension
    print(vector.ndim)
    
    # Create a matrix
    matrix = tf.constant([[10, 7],
                          [7, 10]])
    print(matrix)
    # Check the number of dimension
    print(matrix.ndim)

    # Create another matrix and define the datatype
    another_matrix = tf.constant([[10., 7.],
                                  [3., 2.],
                                  [8., 9.]], dtype=tf.float16)  # specify the datatype with 'dtype'
    print(another_matrix)
    print(another_matrix.ndim)

    # How about a tensor? (more than 2 dimensions, although, all of the above items are also technically tensors)
    tensor = tf.constant([[[1, 2, 3],
                           [4, 5, 6]],
                          [[7, 8, 9],
                           [10, 11, 12]],
                          [[13, 14, 15],
                           [16, 17, 18]]])
    print(tensor)
    print(tensor.ndim)

    # Create the same tensor with tf.Variable() and tf.constant()
    changeable_tensor = tf.Variable([10, 7])
    unchangeable_tensor = tf.constant([10, 7])
    print(changeable_tensor, unchangeable_tensor)

    changeable_tensor[0].assign(7)
    print(changeable_tensor)

    # Will error (can't change tf.constant())
    # unchangeable_tensor[0].assign(7)
    # print(unchangleable_tensor)

    # Create two random (but the same) tensors
    random_1 = tf.random.Generator.from_seed(42)  # set the seed for reproducibility
    random_1 = random_1.normal(shape=(3, 2))  # create tensor from a normal distribution
    random_2 = tf.random.Generator.from_seed(42)
    random_2 = random_2.normal(shape=(3, 2))

    # Are they equal?
    print(random_1, random_2, random_1 == random_2)

    # Create two random (and different) tensors
    random_3 = tf.random.Generator.from_seed(42)
    random_3 = random_3.normal(shape=(3, 2))
    random_4 = tf.random.Generator.from_seed(11)
    random_4 = random_4.normal(shape=(3, 2))

    # Check the tensors and see if they are equal
    print(random_3, random_4, random_1 == random_3, random_3 == random_4)

    # Shuffle a tensor (valuable for when you want to shuffle your data)
    not_shuffled = tf.constant([[10, 7],
                                [3, 4],
                                [2, 5]])
    # Gets different results each time
    print(tf.random.shuffle(not_shuffled))

    # Shuffle in the same order every time using the seed parameter (won't acutally be the same)
    print(tf.random.shuffle(not_shuffled, seed=42))

    # Shuffle in the same order every time

    # Set the global random seed
    tf.random.set_seed(42)

    # Set the operation random seed
    print(tf.random.shuffle(not_shuffled, seed=42))





    
    