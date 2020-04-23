include Makefile.conf

CFLAGS   += -pthread -O3 -g -std=c++14 -DEIGEN_NO_DEBUG -Wall -Wextra
INCLUDE  += -I../../src/ -I./
LIBS     += 
SRCDIR   = ../../src
OBJDIR   = ../../build

.PHONY: default all clean

default: all
all: snchol_file_driver snchol_neglapl_driver

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/communications.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/communications.o $(OBJDIR)/serialization.o $(OBJDIR)/util.o

DEPS_SNCHOL = snchol_utils.hpp snchol.hpp
OBJ_SNCHOL = snchol_utils.o snchol.o

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(MPICXX) $(CFLAGS) -o $@ -c $< $(INCLUDE)

%.o: %.cpp $(DEPS_SNCHOL) $(DEPS)
	$(MPICXX) $(CFLAGS) -o $@ -c $< $(INCLUDE)

snchol_file_driver: snchol_file_driver.cpp $(OBJ) $(OBJ_SNCHOL) $(DEPS) $(DEPS_SNCHOL)
	$(MPICXX) $(CFLAGS) -o $@ $< $(OBJ) $(OBJ_SNCHOL) $(INCLUDE) $(LIBS)

snchol_neglapl_driver: snchol_neglapl_driver.cpp $(OBJ) $(OBJ_SNCHOL) $(DEPS) $(DEPS_SNCHOL)
	$(MPICXX) $(CFLAGS) -o $@ $< $(OBJ) $(OBJ_SNCHOL) $(INCLUDE) $(LIBS)

clean:
	rm -f *.o $(OBJDIR)/*.o *.o 
	rm -f snchol_file_driver snchol_neglapl_driver