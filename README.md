# Apparel_Attribute_Classification

Using transfer learning to retrain the ImageNet model to identify pattern attributes in apparel and classify them into respective categories.

I have used keras to train the model and flask to create a simple app to publish it on local host and take input from the local disk, with minor changes this app can be hosted on a cloud based platform.

Due to the large size of the training direcotry, I was having trouble adding it to the repository even while using git lfs, I have mailed the training folder in zip file along with the email submission. Please include that folder into this directory after cloning this repository into the local machine.

Below are the steps to get this working

```
cd ~
mkdir Apparel_Classification
git clone hhttps://github.com/VivekPillai/Apparel_Attribute_Classification

```

Copy the data.zip (Attached in mail) file into this directory and unzip it.

Run training script

```
python Keras_mutliclass_train.py
```
Run the flask app

```
python app.py
```
This will start a local server.
Go to the browser at 'http://localhost:3000/'

Upload an apparel image from your machine's local disk
Press Upload.

The json response is the predicted category of the apparel based on its features.
While training the modal gives a validation accuracy of about 71.
This can be improved with better labeled data. There are lots of miss labeled data in the training set.
I tried to manualy clean the data set a little. A proper web scrapper with plenty of images for each 
category can also be prove to be very helpful.
