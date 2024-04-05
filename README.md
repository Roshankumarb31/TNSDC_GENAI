<h1>Brain Tumor Detection</h1>
<p>A machine learning project that can be used to detect if a person has brain tumor or not using the MRI scan images of the brain.</p>
<h2>Table of Contents</h2>
<ul style = "list-style-type: circle">
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#data">Data</a></li>
  <li><a href="#model">Model</a></li>
  <li><a href="#predictions">Predictions</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
</ul>
<h2 id="installation">Installation</h2>
<p>Clone this specific Github repository to your local device. Downlad the Dataset from kaggle using the link provided below. Open the repository and run the <code> model_definition.py</code> file (Make sure to change the paths given in the code to the path where the dataset is downloaded). A model  named "model.h5" will be created in the same directory where model_definitions.py is located.</p>
<h2 id="usage">Usage</h2>
<p>After installing the model, the application can be used in the local server by using Django server. For that, open the terminal and change directory to where the file is located and use the command <code>python manage.py runserver </code></p>
<h2 id="data">Data</h2>
<p>The dataset is downloaded from kaggle.com. Its a vast dataset with over 3000 files, seperated into testing and training sets. In each, there are 4 different folders for "Pituitary Tumor", "Meningioma Tumor", "Glioma Tumor" and finally "No Tumor".</p>
<a href="https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri">The link for the dataset download.</a>
<h2 id="model">Model</h2>
<p>The CNN model is defined using the Sequential API from Keras. It consists of several layers:
<ul style = "list-style-type: circle">
<li> Convolutional layers: These layers extract features from the input images.</li>
<li> MaxPooling layers: These layers downsample the feature maps.</li>
<li> Dropout layers: These layers prevent overfitting by randomly dropping a fraction of the connections.</li>
<li> Flatten layer: This layer flattens the feature maps into a 1-dimensional vector.</li>
<li> Dense layers: These layers are fully connected and perform classification.</li>
<li> Output layer: This layer has 4 units (corresponding to the 4 tumor classes) and uses the softmax activation function for multi-class classification.</li>
</ul>
</p>
<h2 id="predictions">Predictions</h2>
<p>In the <code>predictions.py</code>  The code starts by importing necessary libraries and setting up the required variables. It loads image data from different folders for training and testing, resizes the images to a specified size (200x200), and stores them in img_data. Corresponding labels are stored in img_labels. </p>
<p>The image data and labels are shuffled using shuffle() from sklearn.utils to ensure a random order of data. The code then converts the images to grayscale, applies edge detection using Canny algorithm, and reshapes the resulting edge maps. The preprocessed data is stored in newData.</p>
<p>The preprocessed data and labels are split into training and testing sets using train_test_split() from sklearn.model_selection.</p>
<p>The labels are converted from categorical strings to numerical values using the index of the labels in the labels list. The numerical labels are further converted to one-hot encoded vectors using to_categorical() from tf.keras.utils.</p>
<p> The model is compiled with the categorical cross-entropy loss function, Adam optimizer, and accuracy as the evaluation metric. The model is then trained using the training data for a specified number of epochs, with a validation split of 0.1. After training, the model is saved to a file named model.h5.</p>
