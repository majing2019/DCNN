import tensorflow as tf

def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)

class DCNN():
    def __init__(self, batch_size, sentence_length, num_filters, embed_size, top_k, k1, ws, num_hidden, num_class, vocabulary):
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.num_filters = num_filters
        self.embed_size = embed_size
        self.top_k = top_k
        self.k1 = k1
        self.num_hidden = num_hidden

        self.W1 = init_weights([ws[0], embed_size, 1, num_filters[0]], "W1")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[num_filters[0], embed_size]), "b1")

        # 增加int()
        # W2 = init_weights([ws[1], int(embed_dim/2), num_filters[0], num_filters[1]], "W2")
        self.W2 = init_weights([ws[1], int(embed_size), num_filters[0], num_filters[1]], "W2")
        self.b2 = tf.Variable(tf.constant(0.1, shape=[num_filters[1], embed_size]), "b2")

        # 增加int
        # Wh = init_weights([int(top_k*embed_dim*num_filters[1]/4), num_hidden], "Wh")
        self.Wh = init_weights([int(top_k * embed_size * num_filters[1] / 2), num_hidden], "Wh")
        self.bh = tf.Variable(tf.constant(0.1, shape=[num_hidden]), "bh")

        self.Wo = init_weights([num_hidden, num_class], "Wo")

        self.sent = tf.placeholder(tf.int64, [None, sentence_length])
        self.y = tf.placeholder(tf.float64, [None, num_class])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")

        with tf.name_scope("embedding_layer"):
            W = tf.Variable(tf.random_uniform([len(vocabulary), embed_size], -1.0, 1.0), name="embed_W")
            sent_embed = tf.nn.embedding_lookup(W, self.sent)
            self.input_x = tf.expand_dims(sent_embed, -1) # [batch_size, sentence_length, embed_dim, 1]
        
        conv1 = self.per_dim_conv_layer(self.input_x, self.W1, self.b1)
        conv1 = self.k_max_pooling(conv1, self.k1)
        #conv1 = self.fold_k_max_pooling(conv1, k1)
        conv2 = self.per_dim_conv_layer(conv1, self.W2, self.b2)
        top_k = self.top_k
        fold = self.fold_k_max_pooling(conv2, top_k)
        #增加一个int
        #fold_flatten = tf.reshape(fold, [-1, int(top_k * self.embed_size * self.num_filters[1] / 4)])
        fold_flatten = tf.reshape(fold, [-1, int(top_k * self.embed_size * self.num_filters[1] / 2)])
        #fold_flatten = tf.reshape(fold, [-1, int(top_k*100*14/4)])
        # print(fold_flatten.get_shape())
        self.out = self.full_connect_layer(fold_flatten, self.Wh, self.bh, self.Wo, self.dropout_keep_prob)

    def per_dim_conv_k_max_pooling_layer(self, x, w, b, k):
        self.k1 = k
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        with tf.name_scope("per_dim_conv_k_max_pooling"):
            for i in range(self.embed_size):
                conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i])
                #conv:[batch_size, sent_length+ws-1, num_filters]
                conv = tf.reshape(conv, [self.batch_size, self.num_filters[0], self.sentence_length])#[batch_size, sentence_length, num_filters]
                values = tf.nn.top_k(conv, k, sorted=False).values
                values = tf.reshape(values, [self.batch_size, k, self.num_filters[0]])
                #k_max pooling in axis=1
                convs.append(values)
            conv = tf.stack(convs, axis=2)
        #[batch_size, k1, embed_size, num_filters[0]]
        #print conv.get_shape()
        return conv

    def per_dim_conv_layer(self, x, w, b):
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        with tf.name_scope("per_dim_conv"):
            for i in range(len(input_unstack)):
                #yf = input_unstack[i]
                conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i])#[batch_size, k1+ws2-1, num_filters[1]]
                convs.append(conv)
            conv = tf.stack(convs, axis=2)
            #[batch_size, k1+ws-1, embed_size, num_filters[1]]
        return conv

    #增加的函数，只用来做folding操作
    def k_max_pooling(self, x, k):
        input_unstack = tf.unstack(x, axis=2)
        out = []
        with tf.name_scope("k_max_pooling"):
            for i in range(len(input_unstack)):
                conv = tf.transpose(input_unstack[i], perm=[0, 2, 1])
                values = tf.nn.top_k(conv, k, sorted=False).values
                values = tf.transpose(values, perm=[0, 2, 1])
                out.append(values)
            fold = tf.stack(out, axis=2)
        return fold

    def fold_k_max_pooling(self, x, k):
        input_unstack = tf.unstack(x, axis=2)
        out = []
        with tf.name_scope("fold_k_max_pooling"):
            for i in range(0, len(input_unstack), 2):
                fold = tf.add(input_unstack[i], input_unstack[i+1])#[batch_size, k1, num_filters[1]]
                conv = tf.transpose(fold, perm=[0, 2, 1])
                values = tf.nn.top_k(conv, k, sorted=False).values #[batch_size, num_filters[1], top_k]
                values = tf.transpose(values, perm=[0, 2, 1])
                out.append(values)
            fold = tf.stack(out, axis=2)#[batch_size, k2, embed_size/2, num_filters[1]]
        return fold

    def full_connect_layer(self, x, w, b, wo, dropout_keep_prob):
        with tf.name_scope("full_connect_layer"):
            h = tf.nn.tanh(tf.matmul(x, w) + b)
            h = tf.nn.dropout(h, dropout_keep_prob)
            o = tf.matmul(h, wo)
        return o
