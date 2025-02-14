# TDS Project 1

## Overview
This project is part of the IITM BS in Data Science curriculum. The main focus of this project is to demonstrate the use of an agent for handling and automating tasks. Podman is used to containerize and run the application.

## Prerequisites
- Podman installed on your machine

## Getting Started

### Building the Docker Image
To build the Docker image, navigate to the project directory and run the following command:

```sh
podman build -t tds-project1 .
```

### Running the Docker Container
Once the image is built, you can run the container using:

```sh
podman run -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 $IMAGE_NAME
```

This will start the application and map port 8000 of the container to port 8000 on your host machine.

## Accessing the Application
After running the container, the application should be accessible at `http://localhost:8000`.

## Stopping the Container
To stop the running container, use the following command:

```sh
podman stop <container_id>
```

Replace `<container_id>` with the actual container ID, which you can find using:

```sh
podman ps
```

## Cleaning Up
To remove the stopped container and the image, use the following commands:

```sh
podman rm <container_id>
podman rmi tds-project1
```