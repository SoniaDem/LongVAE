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


def get_args(path):
    """
    This method retrieves the arguments from a txt file to use.
    (Stolen with permission from Joe's EdgeGraph code).

    :param path: The path to the txt file.
    :type path: str
    :return params: A dictionary containing the parameters.
    """

    f = open(path, 'r')
    lines = f.readlines()

    # Remove new line token
    lines = [line.strip('\n') for line in lines]

    # Collect parameters in a dictionary.
    params = {}
    for line in lines:
        line_split = line.split(" ")
        # If there is a list specified, then we need to convert it from a long string to a list.
        params[line_split[0]] = line_split[1] if len(line_split) == 2 else [l.strip(',') for l in line_split[1:]]

    return params


def list_to_str(the_list):
    return ' '.join([str(l) for l in the_list])
