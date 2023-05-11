# Malignant Skin Lesion Diagnosis Using Convolutional Neural Networks

How to run this model: 

## Running the model: 
* ssh first
* run: ```source /course/cs1470/cs1470_env/bin/activate```
* run: ``` cd /home/<cs login>/~/dlfinal```

### Rationale: 
Most GPU's we have access to won't be able to run this model with the given preprocessed data. As such, we use the following commands to access Brown CS's GPU stack: 
#### Running on the stack: 
* ```qsub -l gpus=1 -l gmem=24 -o /home/<cs login>/~/dlfinal/out.txt -e  /home/<cs login>/~/dlfinal/error.txt run.sh ```
* Make sure that each task you feed to the stack has a different output file

* ```tail <txt file>``` to check out the results of the task
* ```qstat``` to see the tasks being run
* ```qdel <number>``` to delete a task

## About the model/preprocessing:
* Preprocessing was done using PIL to resize all images to 256x256.
* Images were rotated and color inverted to augment the data.
* These were fed into a ResNet50 block, with which we used transfer learning (setting ```layer.trainable = False```).
* The result of the convolutions was sent into 4 trainable dense layers with some dropout layers imbetween. 

### Contributions:
* All code was written on two computers using group/pair programming techniques. 
* All data, specifically images and pickled files, were stored on the CS department's machines to offload the burden on personal devices. 
* All commented code shows our workflow and things that we tried but ended up not going with due to performance or execution issues.

