CC = g++
CFLAGS = -Wall -O2 -std=c++14

default: main

main: main.o basics.o preprocessing.o Helper.o InvRefiner.o Solver.o InvEncoder.o
	$(CC) $(CFLAGS) -o main main.o basics.o preprocessing.o Helper.o InvRefiner.o Solver.o InvEncoder.o

main.o: main.cpp Solver.h
	$(CC) $(CFLAGS) -c main.cpp

basics.o: basics.cpp basics.h
	$(CC) $(CFLAGS) -c basics.cpp

preprocessing.o: preprocessing.cpp preprocessing.h
	$(CC) $(CFLAGS) -c preprocessing.cpp

Helper.o: Helper.cpp Helper.h
	$(CC) $(CFLAGS) -c Helper.cpp

InvRefiner.o: InvRefiner.cpp InvRefiner.h
	$(CC) $(CFLAGS) -c InvRefiner.cpp

Solver.o: Solver.cpp Solver.h
	$(CC) $(CFLAGS) -c Solver.cpp

InvEncoder.o: InvEncoder.cpp InvEncoder.h
	$(CC) $(CFLAGS) -c InvEncoder.cpp

clean:
	rm *.o main