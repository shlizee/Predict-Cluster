import timeit
from sklearn.neighbors import KNeighborsClassifier
from preprocess.ntu_preprocess import preprocess_pipeline, mini_batch, mini_batch_classify
from model.seq2seq import get_session, Seq2SeqModel, get_feature
import tensorflow as tf
import numpy as np

max_seq_len = 75
rnn_size = 1024
input_size = 75
batch_size = 64
lr = .0001
train_keep_prob = 1.0
iterations = 10000


base_path = "/home/neuralnet/NTU_60/"  # "/home/neuralnet/NW_UCLA/" #
tr_path = "trans_train_data.pkl"
te_path = "trans_test_data.pkl"

# Preprocess the dataset(NTU here for demo) to generate data for
# Step1: seq2seq unsupervised training
# Step2: classification
dsamp_train, dsamp_test, \
fea, lab, seq_len_new, \
test_fea, test_lab, test_seq_len_new = preprocess_pipeline(base_path, tr_path, te_path, mode="cross_subject_data",
                                                     dsamp_frame=50)


# Building Seq2Seq Model applying (Fixed-State or Fixed Weight strategy, modifying loop_fn in Seq2SeqModel to switch
# to fixed-weight strategy, default is fixed-state strategy.)
tf.reset_default_graph()
sess = get_session()

model = Seq2SeqModel(max_seq_len, input_size, rnn_size, batch_size, lr, train_keep_prob)
sess = get_session()
sess.run(tf.global_variables_initializer())


start_time = timeit.default_timer()
knn_score = []
train_loss_li = []
max_score = 0.0

# Training
for i in range(1,iterations+1):
    encoder_inputs, decoder_inputs, seq_len_enc = mini_batch(dsamp_train, seq_len=50, input_size=75, batch_size=256)
    _,gradient_norm,train_loss = model.step(sess, encoder_inputs, decoder_inputs,len(encoder_inputs), seq_len_enc, False)

    if i % 100 == 0:
        print("step {0}:  gradient norm:{1},train loss:{2:.4f}".format(i, gradient_norm, train_loss)) #, f_loss , adversarial loss: {3}
        train_loss_li.append(train_loss)
        end_time = timeit.default_timer()
        print("iteration {}:".format(i),end='')
        print(" using {:.2f} sec".format(end_time-start_time))
        start_time = end_time

    if i % 200 == 0:
        knn_feature = []
        for encoder_inputs, labels, seq_len_enc in mini_batch_classify(fea, lab, seq_len_new, batch_size=64):
            result = get_feature(model,sess,encoder_inputs,len(encoder_inputs),seq_len_enc,labels)
            knn_feature.append(result)
        #     knn_feature.append(np.array(encoder_inputs)[:, 0,:])
        test_knn_feature = []
        for encoder_inputs, labels, seq_len_enc in mini_batch_classify(test_fea, test_lab, test_seq_len_new, batch_size=64):
            result = get_feature(model,sess,encoder_inputs,len(encoder_inputs),seq_len_enc,labels)
            test_knn_feature.append(result)
        knn_feature = np.vstack(knn_feature)
        test_knn_feature = np.vstack(test_knn_feature)
        neigh = KNeighborsClassifier(n_neighbors=9, metric='cosine')
        neigh.fit(knn_feature,np.argmax(lab,axis=1))
        score = neigh.score(test_knn_feature,np.argmax(test_lab,axis=1))
        knn_score.append(score)
        print(f"knn test score at {i}th iterations: ", score)
        if score > max_score:
            model.saver.save(sess,"/home/neuralnet/skeleton_action_recog/NTU_models/cross_subject/fixed_state/lastenc_l1",
                             global_step=i)
            max_score = score
            print("Current KNN Max Score is {}".format(max_score))

    if i % 1000 == 0:
        sess.run(model.learning_rate_decay_op)


