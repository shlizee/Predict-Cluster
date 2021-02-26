import pickle, math, os
import numpy as np
import h5py

def load_data(path):
    """

    :param path:
    :return:
    """
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def normalize_video(video):
    """

    :param video:
    :return:
    """
    max_75 = np.amax(video, axis=0)
    min_75 = np.amin(video, axis=0)
    max_x = np.max([max_75[i] for i in range(0,75,3)])
    max_y = np.max([max_75[i] for i in range(1,75,3)])
    max_z = np.max([max_75[i] for i in range(2,75,3)])
    min_x = np.min([min_75[i] for i in range(0,75,3)])
    min_y = np.min([min_75[i] for i in range(1,75,3)])
    min_z = np.min([min_75[i] for i in range(2,75,3)])
    norm = np.zeros_like(video)
    for i in range(0,75,3):
        norm[:,i] = 2*(video[:,i]-min_x)/(max_x-min_x)-1
        norm[:,i+1] = 2*(video[:,i+1]-min_y)/(max_y-min_y)-1
        norm[:,i+2] = 2*(video[:,i+2]-min_z)/(max_z-min_z)-1
    return norm


def downsample(data, target_frame=50):
    """
    Downsample input data into number of target frames
    :param data:
    :param target_frame:
    :return:
    """
    dsamp = []
    for i in range(len(data)):
        val = np.asarray(data[i]['input'])
        if val.shape[0] > target_frame:
            new_val = np.zeros((target_frame, 75))
            diff = math.floor(val.shape[0] / target_frame)
            idx = 0
            for i in range(0, val.shape[0], diff):
                new_val[idx, :] = val[i, :]
                idx += 1
                if idx >= target_frame:
                    break
            dsamp.append(new_val)
        else:
            dsamp.append(val)
    return dsamp


def preprocess_pipeline(base_path, train_path, test_path, train_name, test_name, mode="cross_subject_data", dsamp_frame=50):
    """
    Generating data separately for training unsupervised seq2seq model & doing classification
    :param base_path:
    :param train_path:
    :param test_path:
    :param mode: cross_subject_data or cross_view_data
    :param dsamp_frame: how many frames/sample for dataset
    :return:
    """
    train_data = load_data(os.path.join(base_path, mode+"/"+train_path))
    test_data = load_data(os.path.join(base_path, mode + "/" + test_path))
    print("Size of training data: ", len(train_data))
    print("Size of test data: ", len(test_data))
    print("Start Normalizing Bones of performers ----")
    # # Normalize Bones
    # for i in range(len(train_data)):
    #     train_data[i]['input'] = normalize_bone(np.array(train_data[i]['input']))
    # for i in range(len(test_data)):
    #     test_data[i]['input'] = normalize_bone(np.array(test_data[i]['input']))

    print("Start Normalizing across all videos ----")
    # Normalize Videos
    for i in range(len(train_data)):
        train_data[i]['input'] = normalize_video(np.array(train_data[i]['input']))
    for i in range(len(test_data)):
        test_data[i]['input'] = normalize_video(np.array(test_data[i]['input']))


    dsamp_train = downsample(train_data, dsamp_frame)
    dsamp_test = downsample(test_data, dsamp_frame)

    print("Start generating data for classification")
    train_path = os.path.join(base_path, train_name)
    test_path = os.path.join(base_path, test_name)
    lab_tr = data_for_classification(train_data, dsamp_train, train_path)
    lab_te = data_for_classification(test_data, dsamp_test, test_path)

    return dsamp_train, dsamp_test, lab_tr, lab_te


def preprocess_pipeline(base_path, train_path, test_path, train_name, test_name, mode="cross_subject_data", dsamp_frame=50):
    """
    Generating data separately for training unsupervised seq2seq model & doing classification
    :param base_path:
    :param train_path:
    :param test_path:
    :param mode: cross_subject_data or cross_view_data
    :param dsamp_frame: how many frames/sample for dataset
    :return:
    """
    train_data = load_data(os.path.join(base_path, mode+"/"+train_path))
    test_data = load_data(os.path.join(base_path, mode + "/" + test_path))
    print("Size of training data: ", len(train_data))
    print("Size of test data: ", len(test_data))
    print("Start Normalizing Bones of performers ----")

    print("Start Normalizing across all videos ----")
    # Normalize Videos
    for i in range(len(train_data)):
        train_data[i]['input'] = normalize_video(np.array(train_data[i]['input']))
    for i in range(len(test_data)):
        test_data[i]['input'] = normalize_video(np.array(test_data[i]['input']))


    dsamp_train = downsample(train_data, dsamp_frame)
    dsamp_test = downsample(test_data, dsamp_frame)

    print("Start generating data for classification")
    train_path = os.path.join(base_path, train_name)
    test_path = os.path.join(base_path, test_name)


    lab_tr = data_for_classification(train_data, dsamp_train, train_path)
    lab_te = data_for_classification(test_data, dsamp_test, test_path)

    return dsamp_train, dsamp_test, lab_tr, lab_te


def data_for_classification(data, dsamp_data, save_name):
    """
    Generating data for classification purpose(After unsupervised training you would do classification)
    :param data:
    :param dsamp_data:
    :return:
    """
    f = h5py.File(save_name, "w")
    fea = []
    lab = []
    seq_len_new = []
    for idx, data in enumerate(data):
        label = data["label"]
        lab = lab + [label+1]
        f.create_dataset(str(idx), data=dsamp_data[idx])
    f.create_dataset('label', data = lab)
    f.close()
    return lab


if __name__ == "__main__":
    base_path = "/NTUProject/" # where you save NTU data
    tr_path = "trans_train_data.pkl"
    te_path = "trans_test_data.pkl"
    train_name = 'NTUtrain_cs.h5' # name save for new data format
    test_name = 'NTUtest_cs.h5'
    dsamp_train, dsamp_test, lab_tr, lab_te = preprocess_pipeline(base_path, tr_path, te_path, train_name, test_name, "cross_subject_data", dsamp_frame=50)
