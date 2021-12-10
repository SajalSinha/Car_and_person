**Aim:**

To build a model that can detect cars and people in any given image. The model should be trained using the given dataset. The dataset follows coco annotations and has 2238 images.
 
**Business Use case:**

We can use this approach to train models for the identification of different items.

**Methodology:**

**Dataset Annotation Conversion** - 

1) The dataset which was given had coco annotations. Such annotations cannot be used as an input to Yolov4 or Yolov5. They both had different input dataset formats. The conversion.py file in GitHub shows how annotations were converted.

2) To evaluate pre-trained models on the test dataset.

3) To use Transfer Learning,i.e to use pre-trained models, to train on our dataset and further compare them. This will usually be the core information for the model to function, with new aspects added to the model to solve a specific task. Programmers will need to identify which areas of the model are relevant to the new task, and which parts will need to be retrained. For example, a new model may keep the processes that allow the machine to identify objects or data, but retrain the model to identify a different specific object. 

_Benefits of Transfer learning:_

_Removing the need for a large set of labelled training data for every new model._

_Improving the efficiency of machine learning development and deployment for multiple models._

_A more generalised approach to machine problem solving, leveraging different algorithms to solve new challenges._

_Models can be trained within simulations instead of real-world environments_

4) And atlast, we will evaluate all models and compare their performance on basis of recall, precision and Mean Average Precision.

**Models used:**

**YOLOV3:**

YOLOv3 (You Only Look Once, Version 3) is a real-time object detection algorithm that identifies specific objects in videos, live feeds, or images. YOLO uses features learned by a deep convolutional neural network to detect an object. Versions 1-3 of YOLO were created by Joseph Redmon and Ali Farhadi.

The first version of YOLO was created in 2016, and version 3, which is discussed extensively in this article, was made two years later in 2018. YOLOv3 is an improved version of YOLO and YOLOv2. YOLO is implemented using the Keras or OpenCV deep learning libraries.
Object classification systems are used by Artificial Intelligence (AI) programs to perceive specific objects in a class as subjects of interest. The systems sort objects in images into groups where objects with similar characteristics are placed together, while others are neglected unless programmed to do otherwise.

**How does YOLOv3 work? (Overview)**

YOLO is a Convolutional Neural Network (CNN) for performing object detection in real-time. CNNs are classifier-based systems that can process input images as structured arrays of data and identify patterns between them (view image below). YOLO has the advantage of being much faster than other networks and still maintains accuracy.

It allows the model to look at the whole image at test time, so its predictions are informed by the global context in the image. YOLO and other convolutional neural network algorithms “score” regions based on their similarities to predefined classes.

High-scoring regions are noted as positive detections of whatever class they most closely identify with. For example, in a live feed of traffic, YOLO can be used to detect different kinds of vehicles depending on which regions of the video score highly in comparison to predefined classes of vehicles.

![image](https://viso.ai/wp-content/uploads/2021/02/YOLOv3-how-it-works.jpg)



**YOLOV5:** 

YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

**Discussion**

**Why two models?**

Comparing two pre-trained models will help us achieve a better understanding and it will be easy to compare overall results.

**Why Yolov3?**

YOLOv3 comes out to be both extremely fast and accurate. In MAP measured at .5 IOU YOLOv3 is on par with Focal Loss but about 4x faster and this is what brought the fast YOLOv2 at par with best accuracies. Moreover, simply just by changing the size of the model you can easily tradeoff between speed and accuracy, hence no retraining required!

YOLOv3 gives a MAP of 57.9 on COCO dataset for IOU 0.5 and the table below shows the comparisons:

![image1](https://github.com/SajalSinha/Car_and_person/blob/main/7b4774b3-8ab1-4099-9b3f-df027520f383.jpg)

YOLOv3 is fast, efficient and has at par accuracy with best two stage detectors (on 0.5 IOU) and this makes it an object detection model that is very powerful. Applications of Object Detection in domains like robotics, retail, manufacturing, media, etc need the models to be very fast keeping in mind a little compromise when it comes to accuracy but YOLOv3 is also very accurate.

This makes it the best model to choose in these kind of applications where speed is important either because:

The products need to be real-time or

The data is just too big.

**Why Yolov5?**

YOLOv5 is implemented in PyTorch initially, it benefits from the established PyTorch ecosystem: support is simpler, and deployment is easier. Moreover as a more widely known research framework, iterating on YOLOv5 may be easier for the broader research community.

YOLOv5 is fast.

YOLOv5 is accurate. In tests on the blood cell count and detection (BCCD) dataset, it achieved roughly 0.895 mean average precision (mAP) after training for just 100 epochs. Admittedly, we saw comparable performance from EfficientDet and YOLOv4, but it is rare to see such across-the-board performance improvements without any loss in accuracy.

YOLOv5 is nearly 90 percent smaller than YOLOv4. This means YOLOv5 can be deployed to embedded devices much more easily.

**Why Not YOLOV4?**

YOLO v5 is nearly 90 percent smaller than YOLO v4. So, it said to be that YOLO v5 is extremely fast and lightweight than YOLO v4, while the accuracy is on par with the YOLO v4 benchmark, and this is the major difference between the two models. So using YoloV3 and YoloV5 for comaprision was made.


**Problems Faced:**

Difficulty with dataset annotation - Since it was in coco we had to change it, and since it was in JSON format we had to parse the file thoroughly and then to get individual file.

YOLOV3 takes a lot of memory and it trains slow as compared to YOLOV5.

Whenever, yolov5 takes input final, so in custom yaml file we have to use class 0,1 for categories rather than 1,2; i.e in our dataset we had two classes 1,2 but yolov5 takes input of category as 0,1 for 2 classes, so we had to change it.

**Results:**

Recall = the share of objects detected by the algorithm;

PrecisionF = the share of ground-truth objects among all the objects that the algorithm predicted correctly;

F1 = 2·Precision×Recall —the harmonic mean of Precision and Recall.

mAP@[.5:.95]: means average mAP over different IoU thresholds, from 0.5 to 0.95, step 0.05 (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95).

**YOLOV5 PERFORMANCE**


![results_yolov5](https://github.com/SajalSinha/Car_and_person/blob/main/yolov5%20results.png)

_**To visit Weight & Bias Dashboard for YOLOV5, click [here](https://wandb.ai/sajalsinha/YOLOv5/runs/31pj9lb8?workspace=user-sajalsinha)**_


**YOLOV3 PERFORMANCE**

![result_yolov3](https://github.com/SajalSinha/Car_and_person/blob/main/Yolov3%20results.png)

_**To visit Weight & Bias Dashboard for YOLOV3, click [here](https://wandb.ai/sajalsinha/YOLOv3/runs/2grnztes?workspace=user-sajalsinha)**_

**Conclusion**

Both the models were trained for 10 epoches. As you can see in the evaluation graphs, YoloV3 performed slightly better than YoloV5 in most of the evaluation metrics but when compared to size and speed then YoloV5 performed well as compared to YoloV3. Due to less number of epoches, we might have got this result, and thier is a high possibility of YoloV5 out-performing YoloV3 when significant number of epoches are used.

