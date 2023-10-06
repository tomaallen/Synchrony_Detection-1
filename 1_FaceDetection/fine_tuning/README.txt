## How to finetune the yolov7?
The code is based on Yolov7 from https://github.com/WongKinYiu/yolov7. "detect_face.py" simply calls the "detect.py" from the official yolov7 package. The latter conducts the frame-wise detection, saves the bounding box txts and example video. After which, detect_face.py reads those txts, compacts them to one csv file. And finally renames the folder and files as you specified.

Actually, I don't think the code can produce good detection results in its current rate, because the yolov7 is finetuned from LEAP data in which the dyadics and lab environments are largely different from your current data. To obtain acceptable result, you will need to manually label the new data and finetune the yolov7 on that again. I would like to instruct you for how to do the fine-tuning.

Before we start, use 
pip install -r requirements.txt
to install all you need.

First, you need to consistantly name and structure your videos. For example, name all your raw videos as
videos/ 
	LP004_STT.mp4
	LP005_STT.mp4
	LP006_STT.mp4

so that your python can iterate them in a forloop. Then, proceed to step_1.py. After which, you should obtain
yolo_samples/
	LP004
	LP005
	LP006

Second, suppose that you have obtained the samples for yolo fituning. Now you will have to label them. To do so, launch YoloLabel.exe from yololabel. Click "Open Files", select yolo_samples/LP004, then select the "obj_names.txt". Now the labeling procedure will start. Draw the bounding box for mom and baby, then press "A"  or "D" for previous or next images, respectively. After which, you should obtain many txt files in your yolo_samples/LP004, they are your labels. Repeat it for all other subjects. Each subject may take you 10-15 mins.

Third, format the data in yolo_samples to feed yolo. To do so, proceed to step_3.py. After which, you should obtain yolo_train.txt and yolo_validate.txt. They will tell yolo what data to use for training and validation.

Forth, now, we will call yolo transfer learning from the command line. I remember you have used yolo for cube/ball detetion in one of LEAP's ML meeting so I suppose you already know what to do.
(The official instruction for this is at https://github.com/WongKinYiu/yolov7#transfer-learning)

Now, in yolov7-main, edit the first three lines from data/leap.yaml, so that the yolo know where the samples are. Just replace the directory for yolo_train.txt and yolo_validate.txt. Leave the test to be the same as val. Save it.

Then, in your conda prompt command line, key in commands to finetune the yolo (You need to login your wandb first. Follow the instruction in the cmd line and make use of google/chatgpt for how to login).

If you want to finetune from the official weight, then key in:

python train.py --workers 1 --device 0 --batch-size 16 --data data/leap.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name isabella_yolo --hyp data/hyp.scratch.custom.yaml --epochs

If you want to finetune from my weight fintuned from LEAP data, then key in:

python train.py --workers 1 --device 0 --batch-size 16 --data data/leap.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'best.pt' --name isabella_yolo --hyp data/hyp.scratch.custom.yaml --epochs 15

Note that epochs can range from 5-20. You need to set it by trial-and-error. You should be able to judge it according to the training logs.

Once done, you will get your new weight in a folder named runs in yolov7-main. When you call detect_face.py, simply set -yolo_weight_path to that new weight.

