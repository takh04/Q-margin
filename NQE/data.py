import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA

def data_load_and_process(dataset, n_features, N_train=None, N_test=None, classes=[0, 1]):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'kmnist':
        kmnist_train_images_path = "kmnist/kmnist-train-imgs.npz"
        kmnist_train_labels_path = "kmnist/kmnist-train-labels.npz"
        kmnist_test_images_path = "kmnist/kmnist-test-imgs.npz"
        kmnist_test_labels_path = "kmnist/kmnist-test-labels.npz"

        x_train = np.load(kmnist_train_images_path)['arr_0']
        y_train = np.load(kmnist_train_labels_path)['arr_0']
        x_test = np.load(kmnist_test_images_path)['arr_0']
        y_test = np.load(kmnist_test_labels_path)['arr_0']
    
    # Normalize images to [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Filter the dataset to only include specified classes
    if len(classes) == 2:
        train_filter_tf = np.where((y_train == classes[0]) | (y_train == classes[1]))
        test_filter_tf = np.where((y_test == classes[0]) | (y_test == classes[1]))
        x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
        x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]

    if isinstance(N_train, int) and isinstance(N_test, int):
        # Balance the classes in the training set
        x_train_balanced = []
        y_train_balanced = []
        for cls in classes:
            idx = np.where(y_train == cls)[0]
            selected_idx = np.random.choice(idx, N_train // 2, replace=False)
            x_train_balanced.append(x_train[selected_idx])
            y_train_balanced.append(y_train[selected_idx])
        
        x_train = np.concatenate(x_train_balanced)
        y_train = np.concatenate(y_train_balanced)

        # Balance the classes in the test set
        x_test_balanced = []
        y_test_balanced = []
        for cls in classes:
            idx = np.where(y_test == cls)[0]
            selected_idx = np.random.choice(idx, N_test // 2, replace=False)
            x_test_balanced.append(x_test[selected_idx])
            y_test_balanced.append(y_test[selected_idx])
        
        x_test = np.concatenate(x_test_balanced)
        y_test = np.concatenate(y_test_balanced)

    
    if n_features is None:
        x_train = np.expand_dims(x_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
        return x_train, x_test, y_train, y_test
    else:
        # Resize images and squeeze dimensions
        x_train = tf.image.resize(x_train[..., np.newaxis], (256, 1)).numpy()
        x_test = tf.image.resize(x_test[..., np.newaxis], (256, 1)).numpy()
        x_train, x_test = tf.squeeze(x_train).numpy(), tf.squeeze(x_test).numpy()

        # Apply PCA for dimensionality reduction
        x_train = PCA(n_features).fit_transform(x_train)
        x_test = PCA(n_features).fit_transform(x_test)

        # Scale features to the range [0, π]
        x_train_scaled, x_test_scaled = [], []
        for x in x_train:
            x_train_scaled.append((x - x.min()) * (np.pi / (x.max() - x.min())))
        for x in x_test:
            x_test_scaled.append((x - x.min()) * (np.pi / (x.max() - x.min())))

        return np.array(x_train_scaled), np.array(x_test_scaled), y_train, y_test