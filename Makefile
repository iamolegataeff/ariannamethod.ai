CC = cc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

.PHONY: all test test-janus janus clean

# ═══ Core AML ═══
all: libaml.a

libaml.a: core/ariannamethod.o
	ar rcs $@ $^

core/ariannamethod.o: core/ariannamethod.c core/ariannamethod.h
	$(CC) $(CFLAGS) -c $< -o $@

# ═══ AML Tests ═══
test: core/test_aml
	./core/test_aml

core/test_aml: core/test_aml.c core/ariannamethod.c core/ariannamethod.h
	$(CC) $(CFLAGS) core/test_aml.c core/ariannamethod.c -o $@ $(LDFLAGS)

# ═══ Janus — first transformer in AML ═══
# "Janus will grow like mycelium, without roots, without a trunk, without a flag."

janus: janus/libjanus.dylib

janus/libjanus.dylib: janus/janus.go janus/lang.go janus/go.mod
	cd janus && go build -buildmode=c-shared -o libjanus.dylib .

test-janus: janus/libjanus.dylib janus/test_janus_c.c
	$(CC) $(CFLAGS) janus/test_janus_c.c -Ljanus -ljanus -o janus/test_janus_c
	cd janus && DYLD_LIBRARY_PATH=. ./test_janus_c

# Full test: AML + Janus basic
test-all: test test-janus

# ═══ Clean ═══
clean:
	rm -f core/*.o core/test_aml libaml.a
	rm -f janus/libjanus.dylib janus/libjanus.h janus/test_janus_c
