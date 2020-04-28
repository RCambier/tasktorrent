include Makefile.conf

CFLAGS   += -pthread -O3 -std=c++14 -DTTOR_SHARED -DNDEBUG -Wall -Wextra
INCLUDE  += -I../../src/
LIBS     += 
SRCDIR   = ../../src
OBJDIR   = ../../build

.PHONY: clean

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/serialization.o $(OBJDIR)/util.o

all: ttor_wait ttor_cholesky ttor_deps omp_wait starpu_wait starpu_deps starpu_deps_stf

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

ttor_wait: ttor_wait.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

ttor_cholesky: ttor_cholesky.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

ttor_deps: ttor_deps.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

omp_wait: omp_wait.cpp
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

starpu_wait: starpu_wait.cpp
	$(CC) $(CFLAGS) $(STARPU_FLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

starpu_deps: starpu_deps.cpp
	$(CC) $(CFLAGS) $(STARPU_FLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

starpu_deps_stf: starpu_deps_stf.cpp
	$(CC) $(CFLAGS) $(STARPU_FLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

clean:
	rm -f ttor_wait ttor_cholesky ttor_deps omp_wait starpu_wait starpu_deps starpu_deps_stf
	rm -f $(OBJDIR)/*.o
