
BINARY = gen-plum.exe

all:
	gcc gen-plum.c -o $(BINARY) -lm


c: clean

clean:
	rm -rf $(BINARY) *~ .*~ *.o


test:
	./$(BINARY) 1 0 
