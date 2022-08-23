import os

def get_class_weights(train_dir):

    # get list of classes from dir names
    # get number of files in each class folder as a list
    n_samples, n_classes = get_counts(train_dir)
    # compute weights ()
    weights = compute_weights(n_samples, n_classes)
    print(f"CLASS WEIGHTS:\n\t{weights}")
    return weights


def compute_weights(samples_num, n_classes):
    weights = []
    idx = []
    total_num = sum(samples_num)

    #compute weight for each class
    for i in range(len(samples_num)):
        idx.append(i)
        tmp_weight = total_num/(n_classes*samples_num[i])
        weights.append(tmp_weight)
        
    weights_dict = dict(zip(idx, weights))
    return weights_dict


def get_counts(train_dir):
    classes = os.listdir(train_dir)
    n_classes = len(classes)

    n_samples = []
    for i in range(n_classes):
        tmp_folder = os.path.join(train_dir, classes[i])
        tmp_num = len(os.listdir(tmp_folder))
        n_samples.append(tmp_num)

    return n_samples, n_classes