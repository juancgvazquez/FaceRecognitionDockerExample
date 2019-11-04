# FaceRecognitionDockerExample
An example of a Docker for cpu with a simple face recognition app  

To build CPU docker, just run : sudo docker-compose up --build  

To build GPU docker, first uncomment the appropiate lines in docker-compose.yml and then run the line above.  
Note: GPU docker requires Nvidia-Docker installed.  

Once built, to run the docker use the following commands:  
GPU:     sudo docker run --gpus all -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/video0 --ipc host face_recognition  
CPU:     sudo docker run -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/video0 --ipc host face_recognition  


Any doubts or comments, reach me at juancgvazquez@gmail.com  

Hope you have fun!
