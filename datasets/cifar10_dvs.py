


import tensorflow_datasets as tfds
import events_tfds.events.cifar10_dvs
#from events_tfds.vis.image import as_frames
#from events_tfds.vis.image import as_frame
from datasets.events.image import as_frames
from datasets.events.image import as_frames_for_nda
from datasets.events.image import as_frame
# from events_tfds.vis.anim import animate_frames



from datasets.augmentation_cifar import cutmix

import tensorflow as tf

import matplotlib.pyplot as plt

from config import config
conf = config.flags

def load():
    #train_ds = tfds.load("cifar10_dvs", split="train", as_supervised=True)


    batch_size = config.batch_size
    num_parallel = tf.data.AUTOTUNE

    if False:
        for events, labels in train_ds:
            print(events)
            print(labels)


        #train_ds = train_ds.map(lambda events, labels: )

    train_ratio = 0.9
    train_ratio_percent = int(train_ratio*100)
    #train_ds, train_ds_info = tfds.load("cifar10_dvs", split="train", as_supervised=True)
    #train_ds = tfds.load("cifar10_dvs", split="train", as_supervised=True)

    #train_ds = tfds.load("cifar10_dvs", split="train[:"+str(train_ratio_percent)+"%]", as_supervised=True, shuffle_files=True)
    #valid_ds = tfds.load("cifar10_dvs", split="train["+str(train_ratio_percent)+"%:]", as_supervised=True)

    train_ds = tfds.load("cifar10_dvs", split="train[10%:]", as_supervised=True, shuffle_files=True)
    valid_ds = tfds.load("cifar10_dvs", split="train[:10%]", as_supervised=True)


    #train_ds = train_ds.map(lambda events, labels: as_frame())


    num_frames = conf.time_step
    conf.time_dim_size = num_frames

    #image_shape = (128,128,3)
    image_shape = (128,128,2)
    #image_shape = (128,128,1)

    # test
    ##for events, labels in train_ds:
    #ds, = train_ds.take(1)
    #events = ds[0]
    #labels = ds[1]
    #as_frames(events,labels,shape=image_shape,num_frames=num_frames)
    #assert False

    #train_ds = train_ds.map(lambda events,labels: as_frame(events,labels,shape=image_shape))

    #
    #cifa10_dvs_img_size = conf.cifar10_dvs_img_size
    #cifar10_dvs_crop_img_size = conf.cifkhh

    # tf tensor version test
    if False:
    #if True:
        for events, labels in train_ds:
            #frames = as_frames(**{k: v.numpy() for k, v in events.items()}, num_frames=20)
            coords = events['coords']
            polarity = events['polarity']
            frame = as_frames(events,labels,shape=image_shape,num_frames=num_frames,augmentation=True)
            #print(labels.numpy())

    # nda (26-09-30): Surro/datasets/cifar10_dvs.py 83-127행을 그대로 옮겼다 (VGGSNN-DVS 레시피).
    # 프레임마다 roll(+-3px) / rotate(+-15도) / shear(+-15도) / cutout(16x16) 중 하나를 1/4 확률로
    # 고른다 (넷 다 안 하는 경우는 없다). 입력 프레임은 as_frames_for_nda 가 resize + 좌우 flip 만 한 것.
    # randaug_en / rand_erase_en 은 Surro 에서도 이 경로가 읽지 않는다 (image_cls 경로 전용).
    # 'nda' 가 아니면 아래 else 는 기존 경로 그대로다 -- 기존 DVS 런과 같은 동작.
    if conf.data_aug_mix == 'nda':
        # tensorflow_addons 는 nda 에서만 필요하다. 기존 경로가 이 import 에 기대지 않도록 안에 둔다.
        import tensorflow_addons as tfa

        @tf.function
        def nda(images, labels):
            def augment_single_frame(img):
                # roll
                def roll_fn():
                    dx = tf.random.uniform([], -3, 4, dtype=tf.int32)
                    dy = tf.random.uniform([], -3, 4, dtype=tf.int32)
                    return tf.roll(img, shift=[dx, dy], axis=[0, 1])

                # rotate
                def rotate_fn():
                    angle = tf.random.uniform([], -15, 15) * 3.141592 / 180.0
                    return tfa.image.rotate(img, angles=angle, interpolation='BILINEAR')

                # shear
                def shear_fn():
                    level = tf.random.uniform([], -15.0, 15.0)
                    rad = level * 3.141592 / 180.0
                    transform = [1.0, tf.math.tan(rad), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    return tfa.image.transform(img, transform, interpolation='BILINEAR')

                # cutout
                def cutout_fn():
                    return tfa.image.random_cutout(img[tf.newaxis, ...], mask_size=(16, 16))[0]

                choice = tf.random.uniform([], 0, 4, dtype=tf.int32)
                img = tf.cond(tf.equal(choice, 0), roll_fn, lambda: img)
                img = tf.cond(tf.equal(choice, 1), rotate_fn, lambda: img)
                img = tf.cond(tf.equal(choice, 2), shear_fn, lambda: img)
                img = tf.cond(tf.equal(choice, 3), cutout_fn, lambda: img)
                return img

            images = tf.map_fn(lambda sample: tf.map_fn(augment_single_frame, sample), images)
            return images, labels

        train_ds = train_ds.map(
            lambda events, labels: as_frames_for_nda(events, labels, dataset_name='CIFAR10_DVS', shape=image_shape, num_frames=num_frames))

        sample = next(iter(train_ds))
        images, labels = sample
        print(f"Shape of the first image: {images.shape}")
        # 배치 뒤에 nda 를 건다 (Surro 와 같은 순서). drop_remainder 는 아래 else 와 같은 이유.
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        train_ds = train_ds.map(lambda images, labels: nda(images, labels))
    else:
        train_ds = train_ds.map(lambda events,labels: as_frames(events,labels,shape=image_shape,num_frames=num_frames,augmentation=True))
    # todo - cutmix
#    if config.flags.data_aug_mix == 'mixup' or config.flags.data_aug_mix == 'cutmix':
#        train_ds_1 = train_ds.map(lambda events,labels: as_frames(events,labels,shape=image_shape,num_frames=num_frames,augmentation=True))
#        train_ds_2 = train_ds.map(lambda events,labels: as_frames(events,labels,shape=image_shape,num_frames=num_frames,augmentation=True))
#        train_ds = tf.data.Dataset.zip((train_ds_1, train_ds_2))
#        #train_ds_num = train_ds_1_num
#
#        train_ds = train_ds.map(
#            lambda train_ds_1, train_ds_2: cutmix(train_ds_1, train_ds_2, dataset_name, input_size, input_size_pre_crop_ratio,
#                                                  num_class, 1.0, input_prec_mode,preprocessor_input),
#            num_parallel_calls=num_parallel)
#    else:
#        #train_ds, train_ds_num = default_load_train()
#        train_ds = train_ds.map(lambda events,labels: as_frames(events,labels,shape=image_shape,num_frames=num_frames,augmentation=True))

    # drop_remainder: 모델 Input 이 batch_size 고정이라 마지막 부분 배치(9000 % 32 = 8)가 들어오면
    # 모양이 안 맞는다. Surro 의 VGGSNN 레시피(batch 32)가 같은 이유로 켠다 (26-09-29).
    # batch 100 이면 9000/1000 이 나누어떨어져 아무것도 안 버린다 (기존 동작 그대로).
    if conf.data_aug_mix != 'nda':   # nda 는 위에서 이미 batch 했다
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(num_parallel)

    #valid_ds = valid_ds.map(lambda events,labels: as_frame(events,labels,shape=image_shape))
    valid_ds = valid_ds.map(lambda events,labels: as_frames(events,labels,shape=image_shape,num_frames=num_frames))
    valid_ds = valid_ds.batch(batch_size, drop_remainder=True)
    valid_ds = valid_ds.prefetch(num_parallel)


    if False: # tfds events code
    #if True: # tfds events code
        for events, labels in train_ds:
            #frames = as_frames(**{k: v.numpy() for k, v in events.items()}, num_frames=20)
            coords = events['coords'].numpy()
            polarity = events['polarity'].numpy()
            frame = as_frame(coords,polarity)
            #print(labels.numpy())
            #print(tf.reduce_max(events["coords"], axis=0).numpy())
            #anim = animate_frames(frames, fps=4)

            print(labels.numpy())
            plt.imshow(frame)

    #valid_ds = train_ds
    # 실제로 도는 표본 수 (drop_remainder 로 버린 만큼 뺀다). 에폭 끝 s_count 를 표본당으로
    # 나누는 분모다 (proc.py spike_count_epoch_end). 예전 식 10000*(1-0.9) 는 999.99... 였다.
    train_ds_num = (int(round(10000*train_ratio)) // batch_size) * batch_size
    valid_ds_num = (int(round(10000*(1-train_ratio))) // batch_size) * batch_size

    return train_ds, valid_ds, valid_ds, train_ds_num, valid_ds_num, valid_ds_num
