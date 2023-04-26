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

# Create Datasets:
testset = create_dataset(json_1)

valid1 = create_dataset(json_2)
valid2 = create_dataset(json_3)

for idx, p in valid2.images.items():
    valid1.images.update({
        idx+100 : valid2.images.get(idx)
    })
    valid1.labels.update({
        idx+100 : valid2.labels.get(idx)
    })

validset = valid1

t1 = create_dataset(json_4)
t2 = create_dataset(json_5)
t3 = create_dataset(json_6)
t4 = create_dataset(json_7)
t5 = create_dataset(json_8)

for idx, p in t2.images.items():
    t1.images.update({
        idx+100 : t2.images.get(idx)
    })
    t1.labels.update({
        idx+100 : t2.labels.get(idx)
    })

for idx, p in t3.images.items():
    t1.images.update({
        idx+200 : t3.images.get(idx)
    })
    t1.labels.update({
        idx+200 : t3.labels.get(idx)
    })

for idx, p in t4.images.items():
    t1.images.update({
        idx+300 : t4.images.get(idx)
    })
    t1.labels.update({
        idx+300 : t4.labels.get(idx)
    })

for idx, p in t5.images.items():
    t1.images.update({
        idx+400 : t5.images.get(idx)
    })
    t1.labels.update({
        idx+400 : t5.labels.get(idx)
    })

trainset = t1

