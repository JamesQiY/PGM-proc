CC=gcc
GCC_OPT = -O2 -Wall -Werror

%.o: %.c
	$(CC) -c -o $@ $< $(GCC_OPT)

run: main
	python3 perfs_student.py

main: very_big_sample.o very_tall_sample.o main.c pgm.c filters.c
	$(CC) $(GCC_OPT) main.c pgm.c filters.c very_big_sample.o very_tall_sample.o -o main.out -lpthread
	

pgm_creator:
	$(CC) $(GCC_OPT) pgm_creator.c pgm.c -o pgm_creator.out
	
clean:
	rm *.o *.out
