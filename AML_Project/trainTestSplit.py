"""
Take the preporcessed images and split them in train and
test data

@author: Dominique Cheray
"""

#necessary imports
import os
import numpy as np
import gzip
from tqdm import tqdm

# load the prepared pieces
f = gzip.GzipFile("positiveFeatures.npy.gz", "r")
pos_feats = np.load(f)
f = gzip.GzipFile("negativeFeatures.npy.gz", "r")
neg_feats = np.load(f)

# create labels
pos_labels = np.ones((pos_feats.shape[0]), dtype=int)
neg_labels = np.zeros((neg_feats.shape[0]), dtype=int)

# stack pos and neg feats and labels together
all_feats = np.vstack((pos_feats, neg_feats))
all_labels = np.hstack((pos_labels, neg_labels))

# determine where to spilt the data (1/3 should be held out for testing)
split_num = int(all_feats.shape[0] * 0.33)
indices = list(range(all_feats.shape[0]))
np.random.shuffle(indices)
train_indices = indices[split_num:]
test_indices = indices[:split_num]

# create files to save wich pieces and labels belong to which set
train_pieces = open("pieces/train_pieces.txt", "w")
test_pieces = open("pieces/test_pieces.txt", "w")
train_labels = open("pieces/train_labels.txt", "w")
test_labels = open("pieces/test_labels.txt", "w")

# now assign all pieces and labels to their respective set
for elem in tqdm(train_indices):
    piece = all_feats[elem]
    f = gzip.GzipFile("pieces/{}.npy.gz".format(elem), "w")
    np.save(file=f, arr=piece)
    f.close()
    train_pieces.write("pieces/{}.npy.gz\n".format(elem))
    train_labels.write("{}\n".format(all_labels[elem]))

for elem in tqdm(test_indices):
    piece = all_feats[elem]
    f = gzip.GzipFile("pieces/{}.npy.gz".format(elem), "w")
    np.save(file=f, arr=piece)
    f.close()
    test_pieces.write("pieces/{}.npy.gz\n".format(elem))
    test_labels.write("{}\n".format(all_labels[elem]))

#close the files
train_pieces.close()
train_labels.close()
test_pieces.close()
test_labels.close()
