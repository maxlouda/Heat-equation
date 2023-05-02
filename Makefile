CppC=g++
CppFLAGS=-O3

OpenMP_CFILES=lodepng.cpp OpenMP.cpp
OpenMP2_CFILES=lodepng.cpp OpenMP2.cpp
Serial_CFILES=lodepng.cpp Serial.cpp
GoodSerial_CFILES=lodepng.cpp GoodSerial.cpp
PThread_CFILES=lodepng.cpp PThread.cpp

all: OpenMP OpenMP2 Serial GoodSerial PThread

OpenMP: $(OpenMP_CFILES)
	$(info Compiling OpenMP!)
	@$(CppC) -fopenmp $(CppFLAGS) $(OpenMP_CFILES) -o openmp
	
OpenMP2: $(OpenMP2_CFILES)
	$(info Compiling OpenMP2!)
	@$(CppC) -fopenmp $(CppFLAGS) $(OpenMP2_CFILES) -o openmp2

Serial: $(Serial_CFILES)
	$(info Compiling Serial!)
	@$(CppC) $(CppFLAGS) $(Serial_CFILES) -o serial

GoodSerial: $(GoodSerial_CFILES)
	$(info Compiling GoodSerial!)
	@$(CppC) $(CppFLAGS) $(GoodSerial_CFILES) -o goodserial

PThread: $(PThread_CFILES)
	$(info Compiling PThread!)
	@$(CppC) $(CppFLAGS) $(PThread_CFILES) -o pthread

clean:
	@rm -rf pics/*.png
	@rm -rf openmp openmp2 serial goodserial pthread


