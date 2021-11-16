# PGM Image Processor
A basic image processor that handles PGM files and implements threads to increase performance


## Description
An image processor that implements pThreads to apply various Laplace filters to the given PGM image for sharpening images. 
Different methods of execution was explored (eg. sharded-rows, work-queue). Done as an assignment for 367 (Parallel Programming)
Python file was included (mostly provided by instructor) to create visual representation of the performance. 

Scinet was used as the environment to understand an accurate represenation of parallel programming. Local results may not reflect the same outcome

usage: recommended usage is written in run job script
