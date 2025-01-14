"Generator")
    model.add(layers.Dense(int(316), activation=tf.nn.relu, name="NoiseToSpatial"))  # 50
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((int(316), 1)))

    conv1d_block(filters=512, upsample=True, index=0)
    conv1d_block(filters=512, upsample=True, index=1)
    conv1d_block(filters=256, upsample=True, index=2)
    conv1d_block(filters=256, upsample=True, index=3)
    conv1d_block(filters=128, upsample=False, index=4)
    conv1d_block(filters=128, upsample=False, index=5)
    conv1d_block(filters=64, upsample=False, index=6)
    conv1d_block(filters=64, upsample=False, index=7)
    conv1d_block(filters=1, upsample=False, activation=tf.nn.tanh, index=8)
    # model.summary()
    # exit(0)
    return model

def define_noise(batch_size_tensor, stddev, z_dim_size):
    with tf.name_scope("LatentNoiseVector"):
        z = tfd.Normal(loc=0.0, scale=stddev).sample(
            sample_shape=(batch_size_tensor, z_dim_size))
        z_perturbed = z + tfd.Normal(loc=0.0, scale=stddev).sample(
            sample_shape=(batch_size_tensor, z_dim_size)) * 1e-5
    return z, z_perturbed

class Discriminator:

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.tail = self._define_tail()
        self.head = self._define_head()

    def _define_tail(self, name="Discriminator"):
        feature_model = tf.keras.models.Sequential(name=name)

        def conv1d_dropout(filters, strides, index=0):
            suffix = str(index)
            feature_model.add(
                layers.Conv1D(filters=filters, strides=strides, name="Conv{}".format(suffix), padding='same',
                              kernel_size=5, activation=tf.nn.leaky_relu))
            feature_model.add(layers.Dropout(name="Dropout{}".format(suffix), rate=0.3))

        conv1d_dropout(filters=32, strides=2, index=5)
        conv1d_dropout(filters=32, strides=2, index=6)
        conv1d_dropout(filters=64, strides=2, index=0)
        conv1d_dropout(filters=64, strides=2, index=1)
        conv1d_dropout(filters=128, strides=2, index=2)
        conv1d_dropout(filters=128, strides=2, index=3)
        conv1d_dropout(filters=256, strides=1, index=4)  # 64
        conv1d_dropout(filters=256, strides=1, index=7)

        feature_model.add(layers.Flatten(name="Flatten"))
        return feature_model

    def _define_head(self):
        head_model = tf.keras.models.Sequential(name="DiscriminatorHead")

        head_model.add(layers.Dense(units=2048, activation='relu'))
        head_model.add(layers.Dropout(rate=0.5))
        head_model.add(layers.Dense(units=2048, activation='relu'))
        head_model.add(layers.Dropout(rate=0.5))
        head_model.add(layers.Dense(units=1024, activation='relu'))
        head_model.add(layers.Dropout(rate=0.5))
        head_model.add(layers.Dense(units=512, activation='relu'))
        head_model.add(layers.Dropout(rate=0.5))

        head_model.add(layers.Dense(units=self.num_classes, activation=None, name="Logits"))
        return head_model

    @property
    def data[0]['dir_input'] = dir_seq[batch_start:batch_end]

        batch_start += batch_size
        # Test data does not use labels
        if data_type == 'test_data':
            yield batch_data[0]
        else:
            yield batch_data


                                                                                                                                                                                                                                                                             stinger-release/attack/varcnn/__pycache__/                                                          0000755 0001762 0000033 00000000000 14737363242 017763  5                                                                                                    ustar   tank06                          sudo                                                                                                                                                                                                                   stinger-release/attack/varcnn/__pycache__/config.cpython-37.pyc                                     0000644 0001762 0000033 00000002110 14737363242 023651  0                                                                                                    ustar   tank06                          sudo                                                                                                                                                                                                                   B
    ��g.  �               @   s<   d dl mZ d dlZeG dd� d��Zedddd d d�ZdS )	�    )�	dataclassNc               @   s�   e Zd ZU eed< eed< eed< eed< eed< dZeed< dZeed	< d
Zeed< d
Zeed< dZ	eed< dZ
eed< dZeed< dZeed< dZeed< dZeed< dZeed< ddggZejed< dS )�Config�num_mon_sites�num_mon_inst_test�num_mon_inst_train�num_unmon_sites_test�num_unmon_sites_trainr   �num_unmon_sitesi�  �
seq_length�	   �	df_epochs�var_cnn_max_epochs�   �var_cnn_base_patienceT�dir_dilations�time_dilations�
inter_timeF�scale_metadataZvar_cnn�
model_name�   �
batch_size�dir�metadata�mixtureN)�__name__�
__module__�__qualname__�int�__annotations__r	   r
   r   r   r   r   �boolr   r   r   r   �strr   r   �t�List� r#   r#   �4/home/tank06/Desktop/stinger/attack/varcnn/config.pyr      s"   
r   �_   i   �d   )r   r   r   r   r   )�dataclassesr   �typingr!   r   �	gb_configr#   r#   r#   r$   �<module>   s                                                                                                                                                                                                                                                                                                                                                                                                                                                           stinger-release/attack/varcnn/__pycache__/model.cpython-37.pyc                                      0000644 0001762 0000033 00000017435 14737363242 023524  0                                                                                                    ustar   tank06                          sudo                                                                                                                                                                                                                   B
    	� gV0  �               @   s�   d dl mZmZmZ d dlmZ d dlmZmZ d dlm	Z	m
Z
mZmZmZmZmZ d dlmZ d dlmZ d dlmZ d dlZd	d
iZddd�Zddd�Zddd�ZG dd� d�ZdS )�    )�ReduceLROnPlateau�EarlyStopping�ModelCheckpoint)�Model)�Conv1D�MaxPooling1D)�Dense�
Activation�ZeroPadding1D�GlobalAveragePooling1D�Add�Concatenate�Dropout)�BatchNormalization)�Adam)�InputN�kernel_initializer�	he_normal�   F��   r   c       	         st   �dkr"� dks|dkrd�nd�� dkr:|r:d� � ��nttd��  ��t|d ��� �������fdd�}|S )	um  A one-dimensional basic residual block with dilations.

    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param dilations: tuple representing amount to dilate first and second conv layers
    Nr   r   �   zb{}�ac                s6  t ��fd��d dd�����d�t��| �}tdd�����d�|�}td	d
�����d�|�}t ��fdd�d d�����d�t��|�}tdd�����d�|�}� dkr�t �df�dd�����d�t��| �}tdd�����d�|�}n| }td�����d�||g�}td	d�����d�|�}|S )N�causalr   Fzres{}{}_branch2a_{})�padding�strides�dilation_rate�use_bias�nameg�h㈵��>zbn{}{}_branch2a_{})�epsilonr   �reluzres{}{}_branch2a_relu_{})r   r   zres{}{}_branch2b_{})r   r   r   r   zbn{}{}_branch2b_{}zres{}{}_branch1_{})r   r   r   zbn{}{}_branch1_{}z
res{}{}_{}zres{}{}_relu_{})r   �format�
parametersr   r	   r   )�x�y�shortcut)�block�
block_char�	dilations�filters�kernel_size�
stage_char�stride�suffix� �3/home/tank06/Desktop/stinger/attack/varcnn/model.py�f)   sB    
zdilated_basic_1d.<locals>.f)r!   �chr�ord�str)	r)   r-   �stager&   r*   �numerical_namer,   r(   r0   r.   )r&   r'   r(   r)   r*   r+   r,   r-   r/   �dilated_basic_1d   s    (r6   c       	         sx   �dkr"� dks|dkrd�nd�d�� dkr>|r>d� � ��nttd��  ��t|d ��� �������fdd	�}|S )
up  A one-dimensional basic residual block without dilations.

    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param dilations: tuple representing amount to dilate first and second conv layers
    Nr   r   r   )r   r   zb{}r   c                s6  t ��fd��d dd�����d�t��| �}tdd�����d�|�}td	d
�����d�|�}t ��fdd�d d�����d�t��|�}tdd�����d�|�}� dkr�t �df�dd�����d�t��| �}tdd�����d�|�}n| }td�����d�||g�}td	d�����d�|�}|S )N�samer   Fzres{}{}_branch2a_{})r   r   r   r   r   g�h㈵��>zbn{}{}_branch2a_{})r   r   r    zres{}{}_branch2a_relu_{})r   r   zres{}{}_branch2b_{})r   r   r   r   zbn{}{}_branch2b_{}zres{}{}_branch1_{})r   r   r   zbn{}{}_branch1_{}z
res{}{}_{}zres{}{}_relu_{})r   r!   r"   r   r	   r   )r#   r$   r%   )r&   r'   r(   r)   r*   r+   r,   r-   r.   r/   r0   q   sB    
zbasic_1d.<locals>.f)r!   r1   r2   r3   )	r)   r-   r4   r&   r*   r5   r,   r(   r0   r.   )r&   r'   r(   r)   r*   r+   r,   r-   r/   �basic_1dV   s    (r8   c          
   C   s<  |dkrddddg}|dkr t }|dkr6dgt|� }tdd| d�| �}tdddd	d
| d�|�}tdd| d�|�}tdd| d�|�}tdddd| d�|�}d}g }xxt|�D ]l\}}	||||ddd	d�|�}x8td|	�D ]*}
|||||
d|
dk�o|| d�|�}q�W |d9 }|�	|� q�W t
d| d�|�}|S )uH  Con training=True)
                    logits_real_unl, features_real_unl = d_model(traces_unl2, training=True)

            with tf.name_scope("GeneratorLoss"):
                feature_mean_real = tf.reduce_mean(features_real_unl, axis=0)
                feature_mean_fake = tf.reduce_mean(features_fake, axis=0)
                # L1 distance of features is the loss for the generator
                loss_g = tf.reduce_mean(tf.abs(feature_mean_real - feature_mean_fake))

            with tf.name_scope(train_scope):
                optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, beta1=0.5)
                train_op = optimizer.minimize(loss_g, var_list=g_model.trainable_variables)

            with tf.name_scope(discriminator_scope):
                with tf.name_scope("Test"):
                    logits_test, _ = d_model(traces_test, training=False)
                    test_accuracy_op = accuracy(logits_test, labels_test)

            with tf.name_scope("Summaries"):
                summary_op = tf.compat.v1.summary.merge([
                    tf.compat.v1.summary.scalar("LossDiscriminator", loss_d),
                    tf.compat.v1.summary.scalar("LossGenerator", loss_g),
                    tf.compat.v1.summary.scalar("ClassificationAccuracyTrain", train_accuracy_op),
                    tf.compat.v1.summary.scalar("ClassificationAccuracyTest", test_accuracy_op)])
            # writer = tf.compat.v1.summary.FileWriter(_next_logdir("tensorboard/wfi10-cw"))

            logger.info("Run training...")
            steps_per_epoch = (len(train_x_labeled_data) + len(
                train_x_unlabeled_data)) // self.batch_size
            steps_per_test = test_x_data.shape[0] // self.batch_size
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                step = 0
                for epoch in range(int(self.train_epochs)):
                    losses_d, losses_g, accuracies = [], [], []
                    logger.info("Epoch {}".format(epoch))
                    pbar = tqdm.trange(steps_per_epoch)

                    sess.run(iterator_labeled.initializer,
                            feed_dict={labeled_X: train_x_labeled_data, labeled_y: train_y_labeled_data})

                    sess.run(iterator_unlabeled.initializer,
                            feed_dict={unlabeled_X: train_x_unlabeled_data, unlabeled_y: train_y_unlabeled_data,
                                        unlabeled_X2: train_x_unlabeled2_data, unlabeled_y2: train_y_unlabeled2_data})

                    logger.info(f"test_x_data shape: {test_x_data.shape}")
                    sess.run(iterator_test.initializer, feed_dict={test_X: test_x_data, test_y: test_y_data})

                    for _ in pbar:
                        if step % 1000 == 0:
                            _, loss_g_batch, loss_d_batch, summ, accuracy_batch = sess.run(
                                [train_op, loss_g, loss_d, summary_op, train_accuracy_op])
                            # writer.add_summary(summ, global_step=step)
                        else:
                            _, loss_g_batch, loss_d_batch, accuracy_batch = sess.run(
                                [train_op, loss_g, loss_d, train_accuracy_op])
                        pbar.set_description("Discriminator loss {0:.3f}, Generator loss {1:.3f}"
                                            .format(loss_d_batch, loss_g_batch))
                        losses_d.append(loss_d_batch)
                        losses_g.append(loss_g_batch)
                        accuracies.append(accuracy_batch)
                        step += 1

                    logger.info("Discriminator loss: {0:.4f}, Generator loss: {1:.4f}, "
                        "Train accuracy: {2:.4f}"
                        .format(np.mean(losses_d), np.mean(losses_g), np.mean(accuracies)))

                    accuracies = [sess.run(test_accuracy_op) for _ in range(steps_per_test)]
                    if np.mean(accuracies) > self.best_acc:
                        self.best_acc = np.mean(accuracies)

                    if epoch == (int(self.train_epochs) - 1):
                        logger.info("Test accuracy: {0:.4f}".format(np.mean(accuracies)))
                        logger.info("Best accuracy: {0:.4f}".format(self.best_acc))
        return self.best_acc                                                                                                                                                                                                                                                stinger-release/attack/varcnn/                                                                      0000755 0001762 0000033 00000000000 14737363242 015553  5                                                                                                    ustar   tank06                          sudo                                                                                                                                                                                                                   stinger-release/attack/varcnn/__init__.py                                                           0000644 0001762 0000033 00000000040 14737363242 017656  0                                                                                                    ustar   tank06 