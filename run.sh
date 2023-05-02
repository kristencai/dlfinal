source /course/cs1470/cs1470_env/bin/activate
cd /home/syu66/CS1470/dlfinal
python3 code/model.py
#% qsub -l gpus=2 -l gmem=24 runme
# qsub -l gpus=1 -l gmem=24 -o /home/syu66/CS1470/dlfinal/out.txt -e  /home/syu66/CS1470/dlfinal/error.txt run.sh 