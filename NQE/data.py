import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA

def data_load_and_process(dataset, n_features, N_train, N_test, classes=[0, 1]):
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
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()

    # Normalize images to [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Filter the dataset to only include specified classes
    if len(classes) == 2:
        train_filter_tf = np.where((y_train == classes[0]) | (y_train == classes[1]))
        test_filter_tf = np.where((y_test == classes[0]) | (y_test == classes[1]))
        x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
        x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]

    # Balance the classes in the training set
    x_train_balanced = []
    y_train_balanced = []
    for cls in classes:
        idx = np.where(y_train == cls)[0]
        selected_idx = np.random.choice(idx, N_train // 2, replace=False)
        x_train_balanced.append(x_train[selected_idx])
        y_train_balanced.append(y_train[selected_idx])
    
    x_train_balanced = np.concatenate(x_train_balanced)
    y_train_balanced = np.concatenate(y_train_balanced)

    # Balance the classes in the test set
    x_test_balanced = []
    y_test_balanced = []
    for cls in classes:
        idx = np.where(y_test == cls)[0]
        selected_idx = np.random.choice(idx, N_test // 2, replace=False)
        x_test_balanced.append(x_test[selected_idx])
        y_test_balanced.append(y_test[selected_idx])
    
    x_test_balanced = np.concatenate(x_test_balanced)
    y_test_balanced = np.concatenate(y_test_balanced)

    
    # Resize images and squeeze dimensions
    if dataset in ['mnist', 'fashion', 'kmnist']:
        x_train_balanced = tf.image.resize(x_train_balanced[..., np.newaxis], (256, 1)).numpy()
        x_test_balanced = tf.image.resize(x_test_balanced[..., np.newaxis], (256, 1)).numpy()
        x_train_balanced, x_test_balanced = tf.squeeze(x_train_balanced).numpy(), tf.squeeze(x_test_balanced).numpy()
    elif dataset == 'cifar10':
        x_train_balanced = tf.image.resize(x_train_balanced, (32, 32)).numpy()
        x_test_balanced = tf.image.resize(x_test_balanced, (32, 32)).numpy()
        x_train_balanced, x_test_balanced = x_train_balanced.reshape(-1, 32*32*3), x_test_balanced.reshape(-1, 32*32*3)

    # Apply PCA for dimensionality reduction
    X_train = PCA(n_features).fit_transform(x_train_balanced)
    X_test = PCA(n_features).fit_transform(x_test_balanced)

    # Scale features to the range [0, π]
    x_train_scaled, x_test_scaled = [], []
    for x in X_train:
        x_train_scaled.append((x - x.min()) * (np.pi / (x.max() - x.min())))
    for x in X_test:
        x_test_scaled.append((x - x.min()) * (np.pi / (x.max() - x.min())))

    return np.array(x_train_scaled), np.array(x_test_scaled), y_train_balanced, y_test_balanced