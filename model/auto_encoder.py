import tensorflow as tf
from .seq2seq import get_session, get_feature
from preprocess.ntu_preprocess import mini_batch_classify
import timeit


def autoencoder(X, inp_dims=2048):
    """
    A simple autoencoder for improving feature representations
    :param X:
    :param inp_dims:
    :return:
    """
    drop = tf.keras.layers.Dropout(rate=0.2)
    FC1 = tf.layers.Dense(units=inp_dims // 2, activation="tanh", name='fc1')
    FC2 = tf.layers.Dense(units=inp_dims // 4, activation="tanh", name='fc2')
    FC3 = tf.layers.Dense(units=inp_dims // 8, activation=None, name='fc3')
    Act = tf.keras.layers.Activation(activation="tanh")
    #     FC4 = tf.layers.Dense(units=inp_dims // 8,activation="tanh",name='fc4')
    FC5 = tf.layers.Dense(units=inp_dims // 4, activation="tanh", name='fc5')
    FC6 = tf.layers.Dense(units=inp_dims // 2, activation=None, name='fc6')
    FC7 = tf.layers.Dense(units=inp_dims, activation=None, name='fc7')
    X = FC1(drop(X))
    X = FC2(drop(X))
    X = FC3(X)
    fea = X
    X_up = Act(X)
    X_up = FC5(X_up)
    X_up = FC6(drop(X_up))
    pred = FC7(drop(X_up))
    return pred, fea


class AE(object):
    """
    An simple vallina nn autoencoder for dimensionality reduction in order to improve the encoder final state
    representation for classification purpose.
    """
    def __init__(self, batch_size, input_size=2048, lr=1e-3, decay_rate=0.999, dtype=tf.float32):
        self.batch_size = batch_size
        self.input_size = input_size
        #         self.lr = lr
        self.lr = tf.Variable(float(lr), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.lr.assign(self.lr * decay_rate)
        with tf.variable_scope("input"):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size], name='enc_in')
        with tf.variable_scope("AE", reuse=tf.AUTO_REUSE):
            pred, self.fea = autoencoder(self.X, self.input_size)
        self.loss = tf.reduce_mean(tf.abs(self.X - pred))

        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.AdamOptimizer(self.lr)
        gradients, self.pred_vars = zip(*opt.compute_gradients(self.loss))
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, 25)
        self.gradient_norms = norm
        self.updates = opt.apply_gradients(zip(clipped_gradients, self.pred_vars), global_step=self.global_step)

    def step(self, session, encoder_final_state, batch_size, train=True):
        if train:
            input_feed = {self.X: encoder_final_state}
            output_feed = [self.updates, self.loss]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1]
        else:
            input_feed = {self.X: encoder_final_state}
            output_feed = [self.loss, self.fea]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1]


def mini_batch_ae(features, batch_size):
    for start in range(0,len(features),batch_size):
        end = min(start+batch_size,len(features))
        yield features[start:end]


def knn_feature_from_seq_model(model, sess, feature, label, seq_len):
    """
    Extract encoder last state as feature for KNN Classifier
    :param model: Trained seq2seq model
    :param sess: trained seq2seq model's session
    :param feature: input sequence
    :param label: action label
    :param seq_len: action sequence length
    :return: A list of encoder last state for classification
    """
    knn_feature = []
    for encoder_inputs, labels, seq_len_enc in mini_batch_classify(feature, label, seq_len, batch_size=64):
        start_time = timeit.default_timer()
        result = get_feature(model, sess, encoder_inputs, len(encoder_inputs), seq_len_enc, labels)
        end_time = timeit.default_timer()
        print(end_time - start_time)
        knn_feature.append(result)
    return knn_feature


if __name__ == "__main__":

    # Code snippet for autoencoder refinement
    # Before that, Loading the seq2seq model and session, get the action sequence and labels of training set.
    knn_feature = knn_feature_from_seq_model(model, sess, fea, lab, seq_len_new)
    tf.reset_default_graph()
    model = AE(batch_size=64)
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    epochs = 100
    # knn_feature = np.vstack(knn_feature)
    for i in range(1,epochs):
        if i % 30 == 0:
            sess.run(model.learning_rate_decay_op)
        loss_sum = 0
        cnt = 0
        for encoder_final_state in mini_batch_ae(knn_feature, batch_size=64):
            _, loss = model.step(sess, encoder_final_state, 64)
            loss_sum += loss
            cnt += 1
        print("reconstruction loss at epoch {} is {}".format(i, loss_sum / cnt))