
Dataset
The Fashion MNIST dataset was used for this project. It consists of 10,000 images across 10 distinct classes, with each class containing 1,000 samples. The dataset is stored in a CSV file, where each row represents a single image and consists of 785 columns—the first column corresponds to the label, while the remaining 784 columns represent pixel values of a 28×28 grayscale image.

Data Preprocessing
Minimal feature engineering was required for this dataset. PyTorch’s Dataset and DataLoader modules were utilized for data loading and batching, ensuring efficient data handling during training and inference.

Model Training & Evaluation
Initially, a basic CNN model was implemented to demonstrate model creation. However, due to its small architecture, the performance was suboptimal.

To improve accuracy, a pre-trained VGG16 model was used, with its final fully connected layer modified to classify the 10 Fashion MNIST classes. Since VGG16 expects input images of size 3×224×224, whereas our dataset consists of 1×28×28 grayscale images, preprocessing steps were applied to resize and convert grayscale images to RGB format.

The training process was straightforward:

Only the last linear layer was retrained.
10 training epochs were used without extensive hyperparameter tuning due to computational constraints.
The model was evaluated using multiple performance metrics:

Accuracy: 0.605
Precision: 0.7342
Recall: 0.6073
F1-score: 0.5644
Model Deployment
A FastAPI-based REST API was developed to expose the model's inference functionality. The /predict endpoint includes basic authentication, with the following credentials:

Username: admin
Password: password123
The API accepts an input array of 784 values (representing a flattened 28×28 grayscale image). Inside the API, the image is converted to an RGB format (3×224×224) to match VGG16’s input requirements.

Containerization & Deployment
The API was containerized using Docker to ensure portability and ease of deployment.
A Docker image was built and successfully executed.
API Testing
The /predict endpoint was validated using Postman, ensuring that the model performs inference correctly.

Postman request screenshot:

<img width="842" alt="image" src="https://github.com/user-attachments/assets/339c71b2-cae1-4a41-943b-70d326ed81d1" />


Model is not been pushed on the github as it was very large file, so i thought to not push it, but the model training code is there to reproduce the same model.


