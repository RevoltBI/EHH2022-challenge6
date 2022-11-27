import tensorflow
import sklearn
import pandas as pd
import random
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Flatten, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta


data = pd.read_parquet("pivoted.pq")

scaler = StandardScaler()
scaler.fit(data)

targets = pd.read_parquet("targets2.pq")
pos_targets = list(targets[targets].dropna().index)
neg_targets = list(targets[~targets].dropna().index)

out = {}
invalid = []
for target in pos_targets + neg_targets:
    try:
        rows = data.loc[target]
    except KeyError:
        invalid.append(target)
        continue
    rows = rows.sort_index()
    r = scaler.transform(rows)
    mask = np.isnan(r)
    r[mask] = 0
    out[target] = np.concatenate([r, (~mask).astype(np.float32)], axis=1)


pos_targets = [x for x in pos_targets if x not in invalid]
neg_targets = [x for x in neg_targets if x not in invalid]

pos_targets_t, pos_targets_v = train_test_split(pos_targets, test_size=0.25, random_state=11112300)
neg_targets_t, neg_targets_v = train_test_split(neg_targets, test_size=0.05, random_state=7334482)


class Collection:
    
    def __init__(self, random, data, indices):
        self.data = data
        self.indices = indices
        self.rnd = random
    
    def _sample(self, idx, length):
        data = self.data[idx]
        sample_length = data.shape[0]
        if sample_length > length:
            data = data[sorted(self.rnd.choices(list(range(data.shape[0])), k=length))]
        elif sample_length < length:
            data = np.concatenate([data, np.zeros([length - sample_length, data.shape[1]])])
        return data
        
    def make_sample(self, length, k):
        rnd = self.rnd
        chosen = rnd.choices(self.indices, k=k)
        return [self._sample(idx, length) for idx in chosen]


class SampleLoader(Sequence):
    def __init__(self, data: pd.DataFrame, pos_targets, neg_targets, epoch_size=600, batch_size=32, seed=22221):
        assert batch_size % 2 == 0
        self.random = random.Random(seed)

        self.pos_col = Collection(self.random, data, pos_targets)
        self.neg_col = Collection(self.random, data, neg_targets)
        self.batch_size = batch_size
        self.half_batch_size = batch_size // 2
        self.epoch_size = epoch_size
        self.labels = np.array([1.0] * self.half_batch_size + [0.0] * self.half_batch_size)
        self.sample_length = 40
        
    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index: int):
        pos_samples = self.pos_col.make_sample(self.sample_length, self.half_batch_size)
        neg_samples = self.neg_col.make_sample(self.sample_length, self.half_batch_size)
        return np.array(pos_samples + neg_samples, dtype=np.float32), self.labels

sampler = SampleLoader(out, pos_targets_t, neg_targets_t)
sampler[3][0][0].shape


class ValLoader(Sequence):
    def __init__(self, data: pd.DataFrame, pos_targets_v, neg_targets_v, epoch_size=3, batch_size=32, seed=22221):
        val_sampler = SampleLoader(out, pos_targets_v, neg_targets_v, epoch_size=epoch_size)
        val_x, val_y = zip(*list(val_sampler))
        self.val_x = val_x
        self.val_y = val_y
        print(val_y)
        self.epoch_size = epoch_size
        
    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index: int):
        return self.val_x[index], self.val_y[index]

val = ValLoader(out, pos_targets_t, neg_targets_t, epoch_size=4)
val[0][0].shape



def make_model():
    inp = Input(shape=(40, 598))
    l1 = Bidirectional(LSTM(128, return_sequences=True))(inp)
    l2 = Bidirectional(LSTM(128, return_sequences=False))(l1)
    flat = Flatten()(l2)
    d1 = Dense(128, activation="relu")(flat)
    d1 = Dropout(0.5)(d1)
    d2 = Dense(1, activation="sigmoid")(d1)
    return Model(inputs=[inp], outputs=[d2])

model = make_model()
fp = tensorflow.keras.metrics.FalsePositives()
fn = tensorflow.keras.metrics.FalseNegatives()
model.compile(loss=BinaryCrossentropy(), optimizer=Adadelta(), metrics=["binary_accuracy", fp, fn])

model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath="models/f{epoch:02d}-{val_loss:.2f}.hdf5",
        save_best_only=True)

model.fit(sampler, validation_data=val, epochs=1000, callbacks=[model_checkpoint_callback])
