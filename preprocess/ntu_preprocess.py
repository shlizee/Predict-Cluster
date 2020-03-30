import pickle, math, os
import numpy as np


def load_data(path):
    """

    :param path:
    :return:
    """
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data


def normalize_bone(video):
    """

    :param video:
    :return:
    """
    bone_connect = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24), (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22), \
                   (0, 16), (16, 17), (17, 18), (18, 19), (0, 12), (12, 13), (13, 14), (14, 15)]
    new_video = np.zeros_like(video)
    new_video[:,0:3] = video[:,0:3]
    #     dif_norm_max = 0.
    #     dif_norm_min = 100.
    #     for bone in bone_connect:
    #         tmp_max = np.amax(np.linalg.norm(video[:,bone[1]*3:bone[1]*3+3]-video[:,bone[0]*3:bone[0]*3+3], axis=1))
    #         tmp_min = np.amin(np.linalg.norm(video[:,bone[1]*3:bone[1]*3+3]-video[:,bone[0]*3:bone[0]*3+3], axis=1))
    #         if tmp_max > dif_norm_max:
    #             dif_norm_max = tmp_max
    #         if tmp_min < dif_norm_min:
    #             dif_norm_min = tmp_min
    for bone in bone_connect:
        dif = video[:,bone[1]*3:bone[1]*3+3]-video[:,bone[0]*3:bone[0]*3+3]
        dif_norm = np.linalg.norm(dif, axis=1)
        new_video[:,bone[1]*3:bone[1]*3+3] = new_video[:,bone[0]*3:bone[0]*3+3]+ dif / np.expand_dims(dif_norm, axis=1) #* np.expand_dims((dif_norm / dif_norm_max),axis=1)
    return new_video


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


def mini_batch(data, seq_len, input_size, batch_size):
    """
    Loading dataset in random batches for Seq2Seq model training
    :param data:
    :param seq_len:
    :param input_size:
    :param batch_size:
    :return:
    """
    encoder_inputs = np.zeros((batch_size, seq_len, input_size), dtype=float)
    seq_len_enc = np.zeros((batch_size,), dtype=float)
    decoder_inputs = np.zeros((batch_size, seq_len, input_size), dtype=float)

    seq_len_dec = np.zeros((batch_size,), dtype=float)
    data_len = len(data)

    for i in range(batch_size):
        index = np.random.choice(data_len)
        data_sel = data[index]
        encoder_inputs[i, :data_sel.shape[0], :] = np.copy(data_sel)
        seq_len_enc[i] = data_sel.shape[0]

        missing_joint = np.random.choice(range(5))
        missing_skels = np.copy(data_sel)
        missing_skels[1:, missing_joint * 4 * 3: (missing_joint * 4 * 3 + 4 * 3)] = np.zeros(
            (data_sel.shape[0] - 1, 4 * 3))
        decoder_inputs[i, :data_sel.shape[0], :] = missing_skels
        seq_len_dec[i] = data_sel.shape[0]
    return encoder_inputs, decoder_inputs, seq_len_enc


def mini_batch_classify(features,labels, seq_len, batch_size):
    """
    Loading dataset for classification purpose
    :param features:
    :param labels:
    :param seq_len:
    :param batch_size:
    :return:
    """
    for start in range(0,len(features),batch_size):
        end = min(start+batch_size,len(features))
        yield features[start:end], labels[start:end], seq_len[start:end]


def preprocess_pipeline(base_path, train_path, test_path, mode="cross_subject_data", dsamp_frame=50):
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
    # Normalize Bones
    for i in range(len(train_data)):
        train_data[i]['input'] = normalize_bone(np.array(train_data[i]['input']))
    for i in range(len(test_data)):
        test_data[i]['input'] = normalize_bone(np.array(test_data[i]['input']))

    print("Start Normalizing across all videos ----")
    # Normalize Videos
    for i in range(len(train_data)):
        train_data[i]['input'] = normalize_video(np.array(train_data[i]['input']))
    for i in range(len(test_data)):
        test_data[i]['input'] = normalize_video(np.array(test_data[i]['input']))


    dsamp_train = downsample(train_data, dsamp_frame)
    dsamp_test = downsample(test_data, dsamp_frame)

    print("Start generating data for classification")
    tr_fea, tr_label, tr_seq_len_new = data_for_classification(train_data, dsamp_train)
    te_fea, te_label, te_seq_len_new = data_for_classification(test_data, dsamp_test)

    return dsamp_train, dsamp_test, tr_fea, tr_label, tr_seq_len_new, te_fea, te_label, te_seq_len_new


def data_for_classification(data, dsamp_data):
    """
    Generating data for classification purpose(After unsupervised training you would do classification)
    :param data:
    :param dsamp_data:
    :return:
    """
    fea = []
    lab = []
    seq_len_new = []
    for idx, data in enumerate(data):
        label = data["label"]
        val = np.asarray(data["input"])
        raw_len = val.shape[0]
        if raw_len > 50:
            seq_len_new.append(50)
            fea.append(dsamp_data[idx])
        else:
            seq_len_new.append(raw_len)
            pad_data = np.zeros((50, 75))
            pad_data[:raw_len, :] = dsamp_data[idx]
            fea.append(pad_data)
        one_hot_label = np.zeros((60,))
        one_hot_label[label] = 1.
        lab.append(one_hot_label)
    return fea, lab, seq_len_new


if __name__ == "__main__":
    base_path = "/home/neuralnet/NTU_60/" #"/home/neuralnet/NW_UCLA/" #
    tr_path = "trans_train_data.pkl"
    te_path = "trans_test_data.pkl"
    dsamp_train, dsamp_test, fea, lab, seq_len_new,\
    te_fea, te_lab, te_seq_len_new = preprocess_pipeline(base_path, tr_path, te_path, mode="cross_view_data", dsamp_frame=50)
    encoder_inputs, decoder_inputs, seq_len_enc = mini_batch(data=dsamp_train, seq_len=75, input_size=75, batch_size=32)