Verified on python 3.8.10 and packages used are NumPy, random, matplotlib and cv2

# Stero-Vision
In this project, we are going to implement the concept of Stereo Vision with 3 different data sets, each of them contains 2 images of the same scenario but taken from two different camera angles. By comparing the information about a scene from 2 vantage points, we can obtain the 3D information by examining the relative positions of objects.

## Steps to run the files:
```
cd Stero-Vision
```
if you want to run the depth estimation on curule dataset
```
python main.py --name curule
```
if you want to run the depth estimation on octagon dataset
```
python main.py --name octagon
```
if you want to run the depth estimation on pendulum dataset
```
python main.py --name pendulum
```

**res** contains the resources files
