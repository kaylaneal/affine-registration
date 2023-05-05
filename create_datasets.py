# LOCAL IMPORTS
from dataset import create_dataset

# Data Paths:
json_1 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (1)/info.json'
images_1 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (1)' 

json_2 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (2)/info.json'
images_2 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (2)' 

json_3 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (3)/info.json'
images_3 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (3)' 

json_4 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (4)/info.json'
images_4 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (4)' 

json_5 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (5)/info.json'
images_5 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (5)' 

json_6 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (6)/info.json'
images_6 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (6)' 

json_7 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (7)/info.json'
images_7 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (7)' 

json_8 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (8)/info.json'
images_8 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (8)' 

testset = create_dataset(json_1)

validset = create_dataset(json_2)

t1 = create_dataset(json_3)
t2 = create_dataset(json_4)
t3 = create_dataset(json_5)
t4 = create_dataset(json_6)
t5 = create_dataset(json_7)
t6 = create_dataset(json_8)

for idx, p in t2.pairs.items():
    t1.pairs.update({
        idx+100 : t2.pairs.get(idx)
    })
    t1.pair_labels.update({
        idx+100 : t2.pair_labels.get(idx)
    })
for idx, p in t3.pairs.items():
    t1.pairs.update({
        idx+200 : t3.pairs.get(idx)
    })
    t1.pair_labels.update({
        idx+200 : t3.pair_labels.get(idx)
    })
for idx, p in t4.pairs.items():
    t1.pairs.update({
        idx+300 : t4.pairs.get(idx)
    })
    t1.pair_labels.update({
        idx+300 : t4.pair_labels.get(idx)
    })
for idx, p in t5.pairs.items():
    t1.pairs.update({
        idx+400 : t5.pairs.get(idx)
    })
    t1.pair_labels.update({
        idx+400 : t5.pair_labels.get(idx)
    })
for idx, p in t6.pairs.items():
    t1.pairs.update({
        idx+500 : t6.pairs.get(idx)
    })
    t1.pair_labels.update({
        idx+500 : t6.pair_labels.get(idx)
    })

trainset = t1
