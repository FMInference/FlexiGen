PREFIX=/usr/local
MANDIR=/usr/man/manl/printenv.l
BINDIR=/usr/local/bin
LIBDIR=/usr/local/lib

CFLAGS += -O2 -Wall -g -pg -D_FILE_OFFSET_BITS=64 -g -fstack-protector-all

all: pagecache-management.so sfr fadv pagecache-management-ignore-reads.so 

pagecache-management-ignore-reads.so: pagecache-management.c Makefile	
	$(CC) -g $(CFLAGS) -shared -fPIC pagecache-management.c -ldl -o pagecache-management-ignore-reads.so -DIGNORE_READS

pagecache-management.so: pagecache-management.c Makefile
	$(CC) -g $(CFLAGS) -shared -fPIC pagecache-management.c -ldl -o pagecache-management.so

clean:
	$(RM) pagecache-management*.so *.o sfr fadv

install: all
	#$(INSTALL) -c -o $(OWNER) -g $(GROUP) -m 755 *.sh $(BINDIR)
	$(INSTALL) -c -m 755 *.sh $(BINDIR)
	$(INSTALL) -c -m 644 *.so $(LIBDIR)
