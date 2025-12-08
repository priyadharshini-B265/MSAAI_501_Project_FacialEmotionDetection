# MSAAI_501_Project_FacialEmotionDetection

Our project aims to develop an AI-based system that can automatically detect multiple faces in an image and classify each person’s facial emotion, providing both individual emotion labels and overall emotion statistics (e.g., “60% happy, 30% neutral, 10% sad”).
The system will combine face detection and emotion classification using deep learning models trained on the AffectNet dataset in YOLO format, capable of handling real-world group images.

Dataset: Facial Expression Image Data AFFECTNET YOLO Format
https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format/data

This project integrates multiple AI concepts, including:

•	Computer Vision (CV): Object detection and facial feature extraction.

•	Deep Learning: CNN architectures, transfer learning, and fine-tuning.

•	Classification: Multi-class emotion classification using neural networks.

•	Search and Optimization: Hyperparameter tuning and model selection.

•	Ethical AI: Addressing dataset bias and ensuring fairness in emotion detection.


The system will:

•	Detect all faces in an uploaded image.

•	Classify each face’s emotion and display a confidence score (e.g., “Happy – 92%”).

•	Handle multiple people with varied expressions and lighting conditions.

•	Display overall emotion percentages for the image (e.g., Happy 60%, Neutral 25%, Sad 15%).

•	Optionally be extended later to support real-time webcam or video input.

Example Output:
Person 1 → Happy (92%)

Person 2 → Neutral (81%)

Person 3 → Surprise (88%)

Overall: Happy 33%, Neutral 33%, Surprise 33%


