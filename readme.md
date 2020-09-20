# Traffic Sign Classification 🚫

Steps:

1. Train, Validation and Test data is in `.p` file so I used `pickle` module to load the data first.

2. Data is in dictionary form. So I need to know the keys to extract my required data

```py
train.keys()
# dict_keys(['coords', 'labels', 'features', 'sizes'])
```

As the keys are known, feature data and target data can be separated using the keys

```py
X_train, y_train = train["features"], train["labels"]
X_valid, y_valid = valid["features"], valid["labels"]
X_test, y_test = test["features"], test["labels"]
```

- totla data: 51839

    - train data: 67.13%

    - test data: 24.36%

    - valid data: 8.51%

3. Before converting to grayscale, data was shuffled using `shuffle` from `sklearn.utils`. Later all the data was normalized. The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values.

4. To build our CNN model we added 2 convolution layer and 2 pooling layer.
    - Pooling helps to reduce the number of extracted features and to avoid overfitting

A dropout layer(20%) was also added to prevent overfitting. Then we flattened the data. And finally 3 dense layer was added.

5. Using `adam` optimizer, the model was compiled.

6. Finally the model was fitted with the training data and a callback function was used to stop the training if `accuracy > 95%`. On the 19th epoch, it reached 95% accuracy.

7. This model was saved using `tensorflow.keras.models`

```py
models.save_model(model, "traffic_sign.hdf5")
```

---

8. For the streamlit app, `st.file_uploader()` is used to upload a picture to classify using this model.

9. Some preprocessing of the uploaded image need to be done to before predicting.
    - reshape the image to (32, 32)
    - convert the colored image to grayscale using `cv2`
    - normalization of the image
    - our model accepts 4D data, so the image need to reshape again.