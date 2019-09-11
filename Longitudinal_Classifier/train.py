def train(data, target):
    """
    Train the Longitudinal model
    :param data: Graph Object from torch-geometric
    :param target: class labels
    :return:
    """

    for d in data: