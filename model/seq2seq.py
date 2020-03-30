import tensorflow as tf
from tensorflow.python.ops.rnn import _transpose_batch_time
from .rnn_wrapper import LinearSpaceDecoderWrapper, ResidualWrapper


class Seq2SeqModel(object):
    def __init__(self, max_seq_len, input_size, rnn_size, batch_size, lr, train_keep_prob, decay_rate=0.95,
                 lambda_a=0.1, lambda_z=0.1, df_size=200, num_class=60, class_lr=1e-3, dtype=tf.float32):
        self.max_seq_len = max_seq_len
        self.rnn_size = rnn_size
        self.df_size = df_size
        self.batch_size = tf.placeholder_with_default(batch_size, shape=())
        self.input_size = input_size
        self.class_lr = tf.Variable(float(class_lr), trainable=False, dtype=dtype)
        self.lr = tf.Variable(float(lr), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.lr.assign(self.lr * decay_rate)
        self.cls_lr_decay = self.lr.assign(self.class_lr * decay_rate)

        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        self.global_step = tf.Variable(0, trainable=False)
        # print('rnn_size = {0}'.format(rnn_size))

        with tf.variable_scope("prediction"):
            with tf.variable_scope("inputs"):
                self.enc_in = tf.placeholder(dtype, shape=[None, self.max_seq_len, input_size], name='enc_in')
                self.dec_in = tf.placeholder(dtype, shape=[None, self.max_seq_len, input_size], name='dec_in')
                self.dec_rel = tf.placeholder(dtype, shape=[None, self.max_seq_len, input_size], name='dec_in')
                self.seq_len = tf.placeholder(tf.int32, [None])
                self.label = tf.placeholder(tf.float32, shape=[None, num_class], name='labels')
                mask = tf.sign(tf.reduce_max(tf.abs(self.enc_in[:, 1:, :]), 2))

            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                cell_fw = [tf.nn.rnn_cell.GRUCell(self.rnn_size // 2) for _ in range(3)]
                cell_bw = [tf.nn.rnn_cell.GRUCell(self.rnn_size // 2) for _ in range(3)]
                ref_outputs, ref_fw_state, ref_bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw,
                                                                                                         cell_bw,
                                                                                                         self.enc_in,
                                                                                                         dtype=tf.float32,
                                                                                                         sequence_length=self.seq_len)
                self.encoder_all_states = ref_outputs
                self.ref_concat = tf.keras.layers.concatenate([ref_fw_state[-1], ref_bw_state[-1]], axis=1)
                self.ref_final_state = self.ref_concat

            pred_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
            cell_ = LinearSpaceDecoderWrapper(pred_cell, self.input_size)
            cell = ResidualWrapper(cell_)
            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                def loop_fn(time, cell_output, cell_state, loop_state):
                    """
                    Loop function that allows to control input to the rnn cell and manipulate cell outputs.
                    :param time: current time step
                    :param cell_output: output from previous time step or None if time == 0
                    :param cell_state: cell state from previous time step
                    :param loop_state: custom loop state to share information between different iterations of this loop fn
                    :return: tuple consisting of
                      elements_finished: tensor of size [bach_size] which is True for sequences that have reached their end,
                        needed because of variable sequence size
                      next_input: input to next time step
                      next_cell_state: cell state forwarded to next time step
                      emit_output: The first return argument of raw_rnn. This is not necessarily the output of the RNN cell,
                        but could e.g. be the output of a dense layer attached to the rnn layer.
                      next_loop_state: loop state forwarded to the next time step
                    """
                    if cell_output is None:
                        # time == 0, used for initialization before first call to cell
                        next_cell_state = self.ref_final_state
                        # the emit_output in this case tells TF how future emits look
                        emit_output = tf.zeros([self.input_size])
                    else:
                        # t > 0, called right after call to cell, i.e. cell_output is the output from time t-1.
                        # here you can do whatever ou want with cell_output before assigning it to emit_output.
                        # In this case, we don't do anything
                        next_cell_state = self.ref_final_state  # cell_state#
                        emit_output = cell_output

                        # check which elements are finished
                    elements_finished = (time >= self.seq_len - 1)
                    finished = tf.reduce_all(elements_finished)

                    # assemble cell input for upcoming time step
                    current_output = emit_output if cell_output is not None else None
                    input_original = self.enc_in[:, 0, :]  # tensor of shape (None, input_dim)

                    if current_output is None:
                        # this is the initial step, i.e. there is no output from a previous time step, what we feed here
                        # can highly depend on the data. In this case we just assign the actual input in the first time step.
                        next_in = input_original
                    else:
                        # time > 0, so just use previous output as next input
                        # here you could do fancier things, whatever you want to do before passing the data into the rnn cell
                        # if here you were to pass input_original than you would get the normal behaviour of dynamic_rnn
                        next_in = current_output

                    next_input = tf.cond(finished,
                                         lambda: tf.zeros([self.batch_size, self.input_size], dtype=tf.float32),
                                         # copy through zeros
                                         lambda: next_in)  # if not finished, feed the previous output as next input

                    # set shape manually, otherwise it is not defined for the last dimensions
                    next_input.set_shape([None, self.input_size])

                    # loop state not used in this example
                    next_loop_state = None
                    return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

                outputs_ta, dec_final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
                dec_outputs = _transpose_batch_time(outputs_ta.stack())
        #                 dec_outputs, dec_final_state = tf.nn.dynamic_rnn(pred_cell, tf.zeros_like(self.dec_in), \
        #                                                 initial_state= self.ref_concat, dtype=tf.float32, \
        #                                                 sequence_length=self.seq_len)

        #                   dec_outputs, dec_final_state = tf.nn.bidirectional_dynamic_rnn(pred_fw_cell, pred_bw_cell, self.dec_in, \
        #                                 initial_state_fw=tf.contrib.rnn.LSTMStateTuple(enc_fw_state[-1].c, enc_fw_state[-1].h), \
        #                                 initial_state_bw=tf.contrib.rnn.LSTMStateTuple(enc_bw_state[-1].c, enc_bw_state[-1].h),
        #                                 dtype=tf.float32, \
        #                                 sequence_length=self.seq_len)

        with tf.variable_scope("pred_fc", reuse=tf.AUTO_REUSE):
            #             FC = tf.layers.Dense(units=75,activation=None,name='pred_skel')
            #             pred_skel = FC(tf.keras.layers.concatenate([dec_outputs[0], dec_outputs[1]], axis = 0))
            #             pred_skel = FC(dec_outputs)
            self.pred_skel = dec_outputs  # pred_skel

        #             self.enc_in[:,1:,:]
        #             print(mask)
        loss_l2 = tf.reduce_sum(tf.abs(self.pred_skel - self.enc_in[:, 1:, :]), 2) * mask
        loss_l2 = tf.reduce_sum(loss_l2, axis=1)
        loss_l2 /= tf.reduce_sum(mask, 1)
        self.loss_pred = tf.reduce_mean(loss_l2)
        # print("loss shape: ", self.loss_pred)


        self.loss = self.loss_pred  # + 0.6*self.f_loss#self.enc_loss

        self.pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "prediction")
        #         params = tf.trainable_variables(self.pred_vars)
        opt = tf.train.AdamOptimizer(self.lr)
        gradients, self.pred_vars = zip(*opt.compute_gradients(self.loss))
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, 25)
        self.gradient_norms = norm
        self.updates = opt.apply_gradients(zip(clipped_gradients, self.pred_vars), global_step=self.global_step)

        with tf.variable_scope("classifier") as scope:
            logits = self.Classifier(self.ref_final_state)
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.label),
                                       name='cost')

        #         Classification learning rate
        optimizer = tf.train.AdamOptimizer(class_lr)
        self.encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "prediction/encoder")
        self.classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classifier")
        # print(self.encoder_vars, self.classifier_vars)
        self.classification_vars = self.encoder_vars + self.classifier_vars
        # print("classifier varibales", self.classification_vars)
        self.train_finetune = optimizer.minimize(self.cost, var_list=self.classification_vars)
        self.train_fixed = optimizer.minimize(self.cost, var_list=self.classifier_vars)

        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        self.pred_label = tf.argmax(logits, 1)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    @staticmethod
    def Classifier(X):
        with tf.variable_scope("classifier_0"):
            Out = tf.layers.dense(X, 60, activation=None)
            return Out

    def step(self, session, encoder_inputs, decoder_inputs, batch_size, seq_len, forward_only, fixed=True, label=None):

        if not forward_only:
            input_feed = {self.enc_in: encoder_inputs, self.dec_in: decoder_inputs, self.seq_len: seq_len,
                          self.batch_size: batch_size}
            output_feed = [self.updates, self.gradient_norms,
                           self.loss]  # ,self.f_loss]#, self.loss_rel]#, self.f_loss]
            outputs = session.run(output_feed, input_feed)
            # session.run(self.reset_op)
            return outputs[0], outputs[1], outputs[2]  # , outputs[3]

        else:
            #             fixed and only train classifier part
            if fixed:
                input_feed = {self.enc_in: encoder_inputs, self.seq_len: seq_len, self.label: label}
                output_feed = [self.train_fixed, self.accuracy, self.pred_label]  # , self.att_weight]
                outputs = session.run(output_feed, input_feed)
                return outputs[0], outputs[1], outputs[2]  # , outputs[3]
            #             Fine tune the encoder together with classifier
            else:
                input_feed = {self.enc_in: encoder_inputs, self.seq_len: seq_len, self.label: label}
                output_feed = [self.train_finetune, self.accuracy, self.pred_label]  # , self.att_weight]
                outputs = session.run(output_feed, input_feed)
                return outputs[0], outputs[1], outputs[2]  # , outputs[3]


def classify_forward(model, session, encoder_inputs, seq_len, label):
    input_feed = {model.enc_in: encoder_inputs, model.seq_len: seq_len, model.label:label}
    output_feed = [model.accuracy, model.pred_label]
    outputs = session.run(output_feed, input_feed)
    return outputs[0], outputs[1]


def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def get_feature(model,session,encoder_inputs,batch_size,seq_len,label):
    """
    Extract encoder last state as feature representation for the action
    :param model:
    :param session:
    :param encoder_inputs:
    :param batch_size:
    :param seq_len:
    :param label:
    :return:
    """
    input_feed = {model.enc_in: encoder_inputs, model.seq_len: seq_len, model.label:label,model.batch_size:batch_size}
    output_feed = [model.ref_final_state]
    outputs = session.run(output_feed, input_feed)
    return outputs[0]


if __name__ == "__main__":
    max_seq_len = 75
    rnn_size = 1024
    input_size = 75
    batch_size = 64
    lr = .0001
    train_keep_prob = 1.0
    iterations = 10000

    tf.reset_default_graph()
    sess = get_session()

    model = Seq2SeqModel(max_seq_len, input_size, rnn_size, batch_size, lr, train_keep_prob)
    sess = get_session()
    sess.run(tf.global_variables_initializer())
