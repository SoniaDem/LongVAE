def ask_dims(image):
    """
    Given an image, this functions asks you to specify the dimensions you would like to plot.
    """
    print(f'\nThe dimensions are: {image.shape}')
    dim0 = int(input('Input dimension 0:\t'))
    dim1 = int(input('Input dimension 1:\t'))
    dim2 = int(input('Input dimension 2:\t'))

    return dim0, dim1, dim2

def print_prog(idx,
               iter_list):
    """
    This function prints the number within a list that we are in in a nice way.
    :param idx:
    :param iter_list:
    :return:
    """
    print(f'[{idx + 1} / {len(iter_list)}]')


def expand_vec(mat, vec):
    """
        As an example, there is a matrix of ones with size (3, 2, 2). We want to multiply
        the vector [1, 2, 3] so we get:
        [[[1  1],
          [1  1]],

         [[2  2],
          [2  2]],

         [[3  3],
          [3  3]]]
    """
    extra_dims = (1,) * (mat.dim() - 1)
    return vec.view(-1, *extra_dims)
