How to run this model: 

ssh first: 
source /course/cs1470/cs1470_env/bin/activate
cd /home/syu66/CS1470/dlfinal


most gpus that we have access to won't be able to run this model with the given 
preprocessed data. as such, we use the following commands to access Brown's 
gpu stack: 
qsub -l gpus=1 -l gmem=24 -o /home/syu66/CS1470/dlfinal/out.txt -e  /home/syu66/CS1470/dlfinal/error.txt run.sh 
make sure that each task you feed to the stack has a different output file

tail <txt file> to check out the results of the task
qstat to see the tasks being run
qdel <number> to delete a task

