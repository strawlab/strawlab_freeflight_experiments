LIB_SRC=contr_fct_TNF_model4.c calc_pathsData.c calculateInput.c dec_fct.c ekf_fct_model4_v0est.c initFunctions.c helpfunctions.c

all: libtnf.so

libtnf.so: $(LIB_SRC)
	$(CC) -msse2 -O3 -Wall -g -fPIC -shared -o $@ $? -lc -lm

clean:
	rm -f *.pyc *.o *.so
	rm -rf build
