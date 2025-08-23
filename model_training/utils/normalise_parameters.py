
def normalise_labels(labels):
    return labels / 150.0 - 0.5


def unnormalise_labels(labels):
    return (labels + 0.5) * 150.0
