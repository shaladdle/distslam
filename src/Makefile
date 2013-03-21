PROGNAME := distslam
OBJECTS := simulator.o slam.o agent.o main.o

all: $(PROGNAME)

distslam: $(OBJECTS)
	g++ $(OBJECTS) -o $@

%.o: %.cpp
	g++ -c $< -o $@

clean:
	rm -rf *.o $(PROGNAME)
