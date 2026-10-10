


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
    # nda_cutmix (26-10-09): 2025-11 코드(duddnr/datasets/cifar10_dvs.py 의 nda_cutmix,
    # 당시 VGGSNN-DVS 81~82%)를 옮겼다. 위 'nda'(2026-03 Surro, 74%)와 다른 점:
    #   (1) roll/rotate/shear 중 하나를 배치 전체(모든 표본, 모든 T 프레임)에 같은 값으로 건다.
    #       프레임마다 따로 걸지 않고, cutout 은 없다.
    #   (2) CutMix 를 항상 건다. 배치 안에서 섞은 짝의 사각형 영역을 붙이고, 라벨은 실제
    #       붙은 면적 비율로 섞는다 (one-hot float -> soft label).
    #   (3) 학습 프레임 resize 가 'nearest' (as_frames_for_nda 의 resize_method).
    # 'nda' 경로는 이 분기와 무관하다 -- 기존 런과의 비교를 위해 한 줄도 바꾸지 않았다.
    elif conf.data_aug_mix == 'nda_cutmix':
        import math
        import tensorflow_addons as tfa

        @tf.function
        def nda_cutmix(images, labels):
            choice = tf.random.uniform([], 0, 3, dtype=tf.int32)

            def flatten_bt(x):
                shape = tf.shape(x)
                return tf.reshape(x, [-1, shape[2], shape[3], shape[4]]), shape

            def unflatten_bt(x_flat, orig_shape):
                return tf.reshape(x_flat, orig_shape)

            def roll_all():
                dx = tf.random.uniform([], -3, 4, dtype=tf.int32)
                dy = tf.random.uniform([], -3, 4, dtype=tf.int32)
                return tf.roll(images, shift=[dx, dy], axis=[2, 3])

            def rotate_all():
                angle = tf.random.uniform([], -15., 15.) * math.pi / 180.0
                flat, shp = flatten_bt(images)
                flat = tfa.image.rotate(flat, angles=angle, interpolation='BILINEAR')
                return unflatten_bt(flat, shp)

            def shear_all():
                level = tf.random.uniform([], -15., 15.)
                rad = level * math.pi / 180.0
                transform = [1.0, tf.math.tan(rad), 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0]
                flat, shp = flatten_bt(images)
                flat = tfa.image.transform(flat, transform, interpolation='BILINEAR')
                return unflatten_bt(flat, shp)

            images = tf.case([(tf.equal(choice, 0), roll_all),
                              (tf.equal(choice, 1), rotate_all)],
                             default=shear_all, exclusive=True)

            B, T, H, W, C = tf.unstack(tf.shape(images))

            # 원본은 tfp.distributions.Beta(1, 1).sample([B]). Beta(1,1) 은 U(0,1) 과 같은 분포라
            # tf.random.uniform 으로 바꿨다. tfp 0.17.0 이 두 서버(venv_1, venv_eip)에 깔려는
            # 있지만 공식 짝은 TF 2.9 다 (우리는 2.12.1) -- 같은 분포를 위해 의존성을 늘리지 않는다.
            # 원본도 alpha=1.0 으로만 불렀다 (duddnr 171행).
            lam = tf.random.uniform([B], 0., 1.)

            idx = tf.random.shuffle(tf.range(B))
            img_shuf = tf.gather(images, idx)
            label_shuf = tf.gather(labels, idx)

            r_x = tf.random.uniform([B], 0, W, tf.int32)
            r_y = tf.random.uniform([B], 0, H, tf.int32)
            r_w = tf.cast(tf.sqrt(1. - lam) * tf.cast(W, tf.float32), tf.int32)
            r_h = tf.cast(tf.sqrt(1. - lam) * tf.cast(H, tf.float32), tf.int32)

            x1 = tf.clip_by_value(r_x - r_w // 2, 0, W)
            y1 = tf.clip_by_value(r_y - r_h // 2, 0, H)
            x2 = tf.clip_by_value(r_x + r_w // 2, 0, W)
            y2 = tf.clip_by_value(r_y + r_h // 2, 0, H)

            # 원본 그대로: 붙일 영역에서 짝 표본 값이 0 인 픽셀은 원래 표본 값을 남긴다
            # (patch != 0 조건). 라벨 비율은 이와 무관하게 사각형 면적으로 정한다.
            def _apply_cutmix(i, img, img_s):
                yy1, yy2, xx1, xx2 = y1[i], y2[i], x1[i], x2[i]
                paddings = [[0, 0],
                            [yy1, H - yy2],
                            [xx1, W - xx2],
                            [0, 0]]
                patch = tf.pad(img_s[:, yy1:yy2, xx1:xx2, :],
                               paddings, 'CONSTANT')
                return tf.where(patch != 0, patch, img)

            mix_imgs = tf.map_fn(
                lambda tup: _apply_cutmix(*tup),
                (tf.range(B), images, img_shuf),
                fn_output_signature=images.dtype)

            lam_area = 1. - (tf.cast((x2 - x1) * (y2 - y1), tf.float32) /
                             tf.cast(H * W, tf.float32))
            lam_area = tf.reshape(lam_area, [-1, 1])

            mix_labels = lam_area * labels + (1. - lam_area) * label_shuf
            return mix_imgs, mix_labels

        train_ds = train_ds.map(
            lambda events, labels: as_frames_for_nda(events, labels, dataset_name='CIFAR10_DVS', shape=image_shape,
                                                     num_frames=num_frames, resize_method='nearest'))
        # 배치 단위 함수라 batch 뒤에 건다 (원본과 같은 순서). drop_remainder 는 아래와 같은 이유.
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        train_ds = train_ds.map(lambda images, labels: nda_cutmix(images, labels))
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
    if conf.data_aug_mix not in ('nda', 'nda_cutmix'):   # nda / nda_cutmix 는 위에서 이미 batch 했다
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
