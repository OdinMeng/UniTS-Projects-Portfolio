# CFLAGS: specify compiler options
CFLAGS = -O3 -fopenmp -march=native

%.o: %.c 
	@echo "Compiling auxiliary code $<"
	gcc ${CFLAGS} -c $<

main: tensorforth.o tensors.o io_tensors.o mystack.o token_interpreter.o tokenizer.o
	@echo "Compiling the main program..."
	gcc ${CFLAGS} -o tensorforth $^
	@echo "Done. Run ./tensorforth <program-name> to run TensorForth. Have Fun!"

.PHONY: clean

clean:
	@echo "Cleaning Up. tensorforth .."
	rm ${wildcard *.o} tensorforth 
