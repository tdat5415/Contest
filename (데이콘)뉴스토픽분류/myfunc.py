import tensorflow as tf

def get_word_list(sequences, minlen=1, maxlen=99):
    words = []
    for seq in sequences:
        words.extend(seq.split())

    words = list(map(lambda x:x.lower(), set(words)))
    words = [word for word in words if minlen <= len(word) <= maxlen]
    words = sorted(words, key=lambda x:len(x), reverse=True)
    return words

def get_len_most_long_seq(sequences):
    temp = list(map(lambda x:x.split(), sequences))
    num_word_list = list(map(len, temp))
    return max(num_word_list)

def data_to_dataset(tra_padded, val_padded, tra_idx, val_idx, weights, num_topic, BUFFER_SIZE=2000, BATCH_SIZE=128):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_topic_ds_list = []
    valid_topic_ds_list = []
    for i in range(num_topic):
        order = tra_idx == i
        tra_ds = tf.data.Dataset.from_tensor_slices((tra_padded[order], tra_idx[order]))
        tra_ds = tra_ds.shuffle(BUFFER_SIZE).repeat()
        order = val_idx == i
        val_ds = tf.data.Dataset.from_tensor_slices((val_padded[order], val_idx[order]))
        val_ds = val_ds.shuffle(BUFFER_SIZE).repeat()
        train_topic_ds_list.append(tra_ds)
        valid_topic_ds_list.append(val_ds)


    train_ds = tf.data.experimental.sample_from_datasets(train_topic_ds_list, weights=weights)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    valid_ds = tf.data.experimental.sample_from_datasets(valid_topic_ds_list, weights=weights)
    valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    return train_ds, valid_ds

class HybridScheduler(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', lr_decay_ratio=0.5, lr_decay_patience=5):
        super(HybridScheduler, self).__init__()
        self.lr_decay_ratio = lr_decay_ratio
        self.lr_decay_patience = lr_decay_patience
        self.best_loss = 99
        self.patience = 1
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        current_loss = logs.get(self.monitor)
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience = 1
        else:
            self.patience += 1
            if self.lr_decay_patience < self.patience:
                lr *= self.lr_decay_ratio
                tf.keras.backend.set_value(self.model.optimizer.lr, lr)
                self.patience = 3
                print('lr : %0.7f'%lr)