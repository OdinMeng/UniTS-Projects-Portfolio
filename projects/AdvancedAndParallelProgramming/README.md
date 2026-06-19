# Advanced and parallel programming projects
This GitHub repository (or .zip file) hosts the projects for the course in "Advanced and Parallel Programming" (2025-2026) held by professor L. Manzoni. This project has been completed in complete autonomy by the student Dino Meng (SM3201466).

The course project is divided in two parts: one in Python, another one in C. 

## C: TensorForth

To compile the program, it is sufficient to invoke `make` in the main folder of the C project. With `make clean` you can clear all of the compiled executables. `examples` contain the examples given in the project guidelines, `my_examples` contain some examples used to benchmark the program and test the correctness of the various implemented functions, and `tests` contain some codes to test individual modules of Tensorforth. Finally, `bash_scripts` contain some bash scripts which can be used to automatically run the official examples and clean up their results afterwards.

After compiling the project, to run the program on a TensorForth script one can simply type the following command:

`./tensorforth <script-name>`

Some interesting notes:

* With `my_examples/make_big.tf` and `my_examples/test_big.tf` you can test the mmap functionality of the program. In particular, it creates a tensor of size 1GB and loads it with mmap. Using valgrind, we can see that we can load large tensors without necessarily loading it into the physical memory

* With some benchmarks done in `my_examples/` and using `hyperfine` to test the execution times (unfortunately I couldn't use profilers such as `stat` as they did not work in my virtual machine), I have found out that in some cases parallelization worsen the performances (such as binary or unary operation on arrays), or improve the performances only starting with certain tensor sizes. So in `tensors.c` I defined some constants to determine the "parallelization cut off values", which are obtained through manual benchmarking. The benchmarks are done with 10 available cores.

* The stack implemented in the project is a dynamically allocated stack, meaning that it is "expanded gradually". It starts with a size of 16, which can be doubled when needed or halved when it's a good moment to do so (in our case when the stack size is the 1/4th of its maximum and the stack had been expanded beforehand). The test file `my_examples/big_stack.tf` tests this feature.

* The TensorForth interpreter is also to generalize the functions with `ndim` greater than 2. The example file `my_examples/3dim.tf` shows an example where `ndim=3`. However, in order to run this script it is necessary to reconfigure the constant `MAX_DIM` in the header file `tensors.h`.

* To run the bash scripts, you have to run them from the main folder. So the working directory has to be `C` and you must simply type `./bash/scripts/<my_script.sh>` to run the script. Note that you might have to add execution permissions to the bash scripts, which can be done by using `chmod +x <filename.sh>` command.

## Python: Renderer

All of the important class implementations are in main folder of the Python project; each `.py` file implements a class, with the exception of `main.py` which is used as the "entry point" of the program. `misc` folder contains a notebook which was used to prototype and test the implmentations. `examples` contains the examples as given in the project descriotion. `my_examples` contain some (or only one) personal example(s) to test the rendering pipeline's edge cases.

To run the renderling pipeline on the project examples type the following command:

`python main.py ./examples/palette.json ./examples/scene.json ./examples/tiles.bin ./examples/sprites.bin test.png`

The required libraries to run the python scrips are detailed in `requirements.txt`

Notes:

* The rendering pipeline will validate the input files strictly as specified in the project guidelines, so for example the `scenes.json` file must contain all of the requested fields (`transparent_index`, `tile_map`, and `sprites`) and their respective types and values must be valid. For example, in `tile_map` only bidimensional matrixes with 20 columns and 15 rows will be considered.

* The only case where a warning is issued is in the `Palette` class, where invalid values such as non-integers or out-of-bound values will be automatically recasted as a `uint8` integer.
