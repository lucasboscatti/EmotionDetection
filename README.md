# Project README

## Lucas Boscatti's Final Project Nero

This project is a facial emotion detection system developed by Lucas Boscatti. It utilizes deep learning techniques to recognize and classify emotions in real-time from images or video sources. The project provides a user-friendly interface for running the emotion detection system with various options and configurations.

### Installation

1. Clone this repository to your local machine:

```
git clone https://github.com/lucasboscatti/EmotionDetection.git
```

2. Install Docker

3. Run the following command to build the Docker image:

```
docker build --tag python-docker .
```

4. Run the following command to run the Docker image:

```
docker run -p 5000:5000 python-docker
```

5. Visit http://localhost:5000 in your web browser to access the emotion detection system.

6. To stop the Docker container, run the following command:

```
docker stop python-docker
```
