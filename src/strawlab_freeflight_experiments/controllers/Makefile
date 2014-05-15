LIB_SRC=calc_pathAndDer.c calculateInput.c contr_fct_subopt_MPC_model2.c dec_fct.c ekf_fct_model2.c helpfunctions.c initFunctions.c sys_cost_fct.c

all: libmpc.so

libmpc.so: $(LIB_SRC)
	$(CC) -Wall -g -fPIC -shared -o $@ $? -lc -lm

testspeed: testspeed.c $(LIB_SRC)
	$(CC) -Wall -g -pg -o $@ $? -lc -lm

clean:
	rm -f *.pyc *.o *.so testspeed
	rm -rf build
