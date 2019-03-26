# Weight Estimation with Stereo Camera
This is the code relative to the CV Project @UNITN.
The main purpose of this work is to estimate the weight of a person situated in front of the camera.
The whole project is based on the concepts of stereo vision.

### Small Example
![simple frame](screen.png)
This example shows how the application works:
the ZED camera acquire the video and then at every frame the program compute the height of the subject, the width and the surface area (of the front view of the person). With these three components the program, then, estimates the weight in KG of the person.

### Requirements
* ZED Stereo Camera
* Python 3
* OpenCV for Python (installed via anaconda with the command `conda install opencv`)
* [CUDA gpu support](https://developer.nvidia.com/cuda-downloads)
* [ZED SDK](https://www.stereolabs.com/developers/)


### Run
In order to run the application we need to write down the following code in the terminal:

```
Python3 test.py person_name weight_in_kg
```
