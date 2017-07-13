from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

def TestCNN(features_name):
    bottleneck_file = 'bottleneck_features\Dog' + features_name + 'Data.npz';
    print(bottleneck_file)
    bottleneck_features = np.load(bottleneck_file)
    train_Xception = bottleneck_features['train']
    valid_Xception = bottleneck_features['valid']
    test_Xception = bottleneck_features['test']

    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))

    Xception_model.add(Dense(5000))
    Xception_model.add(BatchNormalization())
    Xception_model.add(Activation('relu'))
    Xception_model.add(Dropout(0.5))

    Xception_model.add(Dense(133))
    Xception_model.add(BatchNormalization())
    Xception_model.add(Activation('softmax'))
    #Xception_model.summary()

    Xception_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    weight_file_name = 'saved_models/weights.best.' + features_name + '.hdf5'

    checkpointer = ModelCheckpoint(filepath=weight_file_name,
                                   verbose=1, save_best_only=True)

    Xception_model.fit(train_Xception, train_targets,
              validation_data=(valid_Xception, valid_targets),
              epochs=20, batch_size=20, callbacks=[checkpointer], verbose=2)


    Xception_model.load_weights(weight_file_name)

    Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Xception]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/len(Xception_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

print('#'*40)
print('InceptionV3')
print('#'*40)
TestCNN('InceptionV3')

print('#'*40)
print('Resnet50')
print('#'*40)
TestCNN('Resnet50')

print('#'*40)
print('VGG19')
print('#'*40)
TestCNN('VGG19')

print('#'*40)
print('Xception')
print('#'*40)
TestCNN('Xception')