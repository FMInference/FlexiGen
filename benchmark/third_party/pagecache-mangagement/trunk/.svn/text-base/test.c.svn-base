#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>


#ifdef __i386__
#define NR_sync_file_range 314
#endif

#ifdef __x86_64__
#define NR_sync_file_range 277
#endif



       int
       main(int argc, char *argv[])
       {
           pid_t tid;
           tid = (long) syscall(SYS_gettid);
		syscall(NR_sync_file_range, 0, 0, 0, 0);
return 0;
       }
