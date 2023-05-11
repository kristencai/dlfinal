How to run this model: 


# Running the model: 
* ssh first
* run: ```source /course/cs1470/cs1470_env/bin/activate```
* run: ``` cd /home/<cs login>/~/dlfinal```

## Rationale: 
most gpus that we have access to won't be able to run this model with the given preprocessed data. as such, we use the following commands to access Brown's gpu stack: 
### Running on the stack: 
* ```qsub -l gpus=1 -l gmem=24 -o /home/<cs login>/~/dlfinal/out.txt -e  /home/<cs login>/~/dlfinal/error.txt run.sh ```
* make sure that each task you feed to the stack has a different output file

* ```tail <txt file>``` to check out the results of the task
* ```qstat``` to see the tasks being run
* ```qdel <number>``` to delete a task

# About the model/preprocessing:
* Preprocessing was done using PIL to resize all images
* Images were rotated and color inverted
* these were fed into a resnet50 block, with which we used transfer learning
* the result of the convolutions was sent into 4 dense layers to apply transfer learning

