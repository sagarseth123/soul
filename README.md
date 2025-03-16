Dataset
mnist fashion datset is been used,
it containes 10,000 items with 10 classes, each class has 1000 records, the dataset is in csv file where roes present one item and columns contains 785 columns, where 1 columns was label and rest of the columns were image pixel of 28*28 size.

Data Preprocessing:
Not much feature engineering was needed on dataset. for laoding  and batching the data i use pytorch Datasets and dataloaders.

Model Training & Evaluation:
first i tried to train a very simple cnn model to showcase model creation, which definately won't had very bad accuracy as it will very small model.
later i used existing vgg16 model and re-trained it's last linear layer on my dataset, the vgg16 accept the image of format 3*224*224 but our dataset had the images of 1*28*28 so we need to resize out image into 3*224*224 with respect to vgg16 input format. the training was very simple and only 10 epochs is been used to re-train and i didn't experiment much with the training as it was very costly and time consuming step, i took the model after first training itself and started it evaluation,
i checked the model performance using various performance metrics and score for each one of them is:
Accuracy: 0.605
Precision: 0.7342281276775325
Recall: 0.6072630714749905
F1: 0.5644224763060713

Model Deployment:
I created a simple fast API with /predict endpoint with very basic authentication with username = admin and passowrd = password = password123 for performing inferecnce on the model:
the input the api is a array of size 784(28*28) where each element is image pixel, anf later inside the api we are converting it into a rgb image of size 224*224*3.
once the development completed we containerize the api using docker and try to build the image and later ran that image.

here i mention the postman screenshot:
<img width="842" alt="image" src="https://github.com/user-attachments/assets/339c71b2-cae1-4a41-943b-70d326ed81d1" />


