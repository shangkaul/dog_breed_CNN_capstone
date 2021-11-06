# Dog Breed Predictor - CNN_Transfer Learning
The objective of this project is to predict the dog breed from image input. The model used is CNN with transfer features from Resnet%) model.

# Software Requirements
- Python 3.x
- Numpy
- Pandas
- Matplotlib
- Scikit-Learn
- Keras
- Tensorflow
- OpenCV
 
 ## File structure and desciption:
```
root
│   .gitignore
│   CODEOWNERS
│   dog_app-zh.ipynb
│   dog_app.ipynb
│   extract_bottleneck_features.py
│   LICENSE
│   LICENSE.txt
│   README.md
│   test-img1.jpg
│   test-img2.jpg
│   test-img3.jpg
│   test-img4.jpg
│   test-img5.jpg
│   test-img6.jpg
│
├───bottleneck_features
│       .gitignore
│       DogResnet50Data.npz
│
├───haarcascades
│       haarcascade_frontalface_alt.xml
│
├───images
│       American_water_spaniel_00648.jpg
│       Brittany_02625.jpg
│       Curly-coated_retriever_03896.jpg
│       Labrador_retriever_06449.jpg
│       Labrador_retriever_06455.jpg
│       Labrador_retriever_06457.jpg
│       sample_cnn.png
│       sample_dog_output.png
│       sample_human_2.png
│       sample_human_output.png
│       Welsh_springer_spaniel_08203.jpg
│
├───requirements
│       dog-linux-gpu.yml
│       dog-linux.yml
│       dog-mac-gpu.yml
│       dog-mac.yml
│       dog-windows-gpu.yml
│       dog-windows.yml
│       requirements-gpu.txt
│       requirements.txt
│
└───saved_models
        .gitignore
        Resnet50_transfer.h5
        weights.best.from_scratch.hdf5
        weights.best.Resnet50.hdf5
        weights.best.VGG16.hdf5
```
The above tree describes all the files as part of this project.
   - dog_app.ipynb - This is the jupyter notebook that drives the entire code for the project
   - bottleneck_features : Contains the pretrained ResNet50 model for transfer learning.
   - haar cascades: Contains the haarcascade XML file to be used by cascade classifiers for face detection.
   - saved_models: Contains the saved best model weights of pre-trained models.
   - images: Stock images that came with the jupyter barebones code @ Udacity
   - test-img{1-6} : Images to test the model prediction output

## Steps to replicate the project
   - All the dependencies are in requirement.txt and requirements-gpu.txt. A new virtual environment is preferable.
   Using python - preferable version <3.8 as accompanying tensorflow and keras versions not compatible with 3.8.
   - Install python, create a virtual env
   - activate the virtual env, install required packages as mentioned in requirements folder.
   - Run the Jupyter notebook (GPU acceleration preferred)
   - You can update the image path to run the predictor on your input images as well.
   
 ## Conclusion
 The Transfer model achieves a test accuracy of 83.5%. Using transfer learning, training time is reduced significantly without an accuracy trade off. <br>
 I have summarized the results in a blog [Link](https://blogs.shangkaul.in/)<br>
 Next steps include, creating a flask app to use this model to predict dog breeds in real time. This would take some time since I'll have to upgrade the dependencies to more recent versions. Eventually this app would be deployed on - my site https://shangkaul.in/
 
 ## Acknowledgement and Licensing terms
 All course materials are courtesy Udacity - [Terms of Service](https://www.udacity.com/legal)
 
 ## References
 - https://www.udacity.com/
 - https://github.com/udacimak/udacity-HowToDownloadWorkspaces
 - [Transfer Learning Basics](https://www.youtube.com/watch?v=LsdxvjLWkIY)
 - https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
 - https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363
 - https://www.psychosocial.com/article/PR280274/19940/
