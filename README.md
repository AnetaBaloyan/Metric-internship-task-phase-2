# Metric-internship-task-phase-2
### Intro
This is the task for [Metric](https://metric.am/) summer ML internship phase 2. In this project I am 
creating a **Smile Detector**, which will open the webcam, detect when you smile and your both eyes are open, 
and take a shot. 
For this project, I am using *OpenCV* with python to deal with images and frames and *dlib* for classification.

***
### How it works
The tool works for any number of faces and takes a shot only when **everyone** is smiling. Every detected 
face is shown in a rectangle of a distinct color and the corresponding emotion is 
written on the left of the frame with the same color. Eyes are also highlighted, the left and the right
 eyes are outlined with different colors. When an eye is not "open" enough, the outline
 color is dimmed, otherwise it is bright. After taking a shot the process freezes for a second to 
 avoid taking the same shots many times and loading the memory with similar pictures. You can see the 
number of shots taken at toe top left corner. **In order to terminate the process, you need to press 'q'.**
 After termination, you can find two versions of each shot in the *./captured_photos* folder
  in *.jpg* format: a clean one and one with all the outlines and rectangles. 

***
### Testing on a dataset
I have also included a test dataset of 199 1024x1024 images which I took from 
[this sourse](https://github.com/NVlabs/ffhq-dataset). 
You can add your own dataset and test the detector on it by just providing the path in *start.py*. When the
classification is completed, you can find the results in two newly added folders: *[your_test_dir]/good* and 
*[your_test_dir]/bad*. Intuitively, "good shots" are in *./good* and "bad shots" are in *./bad*. All the 
classified pictures have the drawings on them to explain the decision.

***
### Some history of bad attempts
At first, I tried to reach the desired outcome using only 
[haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades), 
however, some details would never pass the threshold. Then, I ditched the haarcascades and switched to **dlib**, 
using [this source](https://github.com/wahid18benz/Detect-Facial-Features) as a guide.
This resulted into major performance increase, as I added explicit checks for eyes being open/closed, 
detection of emotions more than just 'happy'. 
As opposed to what was before, the program also started behaving better on streams with poorer quality. 


***
### Room for improvements
**The drawback**, however, is that in order to evaluate the "openness of the eyes", 
I had to set a fixed threshold, after which an eye is considered "not open enough". 
This might discriminate people with narrower eye structure, as their open eyes may not pass the threshold 
in some cases. This will be especially 
uncomfortable to people whose eyes get very narrow when they smile (like me). So, when
using this tool, please, smile with your eyes wide open (until I figure something out).

Another slight issue is that the detector might interpret a neutral face 
as happy if the lighting is not that good. The reason behind this is that
the shadows sometimes create an illusion of a smile. So, my recommendation is also
to use the tool in a place with good lighting. 


#### Thank you!



Author: [Aneta Baloyan](https://www.linkedin.com/in/aneta-baloyan/) 
