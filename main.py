from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI, File
from PIL import Image
import numpy as np
import uuid
import aiofiles
app = FastAPI()
DATA_DIR = 'data/'
TEST_DIR = 'test/'
DATASET = 'mnist' 
TEST_DATA_FILENAME = DATA_DIR + DATASET + '/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/t10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/train-labels-idx1-ubyte'
k = None
y_train = None
X_train = None

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="OCR",
        version="1.0.0",
        description="OpenAPI para a aplicação de reconhecimento de caracteres (OCR)",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

def read_image(path):
    return np.asarray(Image.open(path).convert('L'))


def write_image(image, path):
    img = Image.fromarray(np.array(image), 'L')
    img.save(path)


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')


def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def dist(x, y):
    """
    Returns the Euclidean distance between vectors `x` and `y`.
    """
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
            for x_i, y_i in zip(x, y)
        ]
    ) ** (0.5)


def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]


def get_most_frequent_element(l):
    return max(l, key=l.count)


def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_sample_idx, test_sample in enumerate(X_test):
        print(test_sample_idx, end=' ', flush=True)
        training_distances = get_training_distances_for_test_sample(
            X_train, test_sample
        )
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1]
            )
        ]
        candidates = [
            y_train[idx]
            for idx in sorted_distance_indices[:k]
        ]
        top_candidate = get_most_frequent_element(candidates)
        y_pred.append(top_candidate)
    print()
    return y_pred

def startup():
    global k
    global y_train
    global X_train
    n_train = 1000
    k = 7
    X_train = read_images(TRAIN_DATA_FILENAME, n_train)
    y_train = read_labels(TRAIN_LABELS_FILENAME, n_train)
    X_train = extract_features(X_train)

startup()

def read_ocr_file(file_path):
    X_test = [read_image(file_path)]
    X_test = extract_features(X_test)
    return knn(X_train, y_train, X_test, k)
   

@app.post("/read/")
async def read_digit(file: bytes = File(...)):
    file_path = f'{DATA_DIR}/{str(uuid.uuid4())}.png'
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        await out_file.write(file)

    return read_ocr_file(file_path)
