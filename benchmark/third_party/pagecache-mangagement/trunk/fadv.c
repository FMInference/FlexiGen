/*
 */

#define _XOPEN_SOURCE 600
#include <fcntl.h>

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <limits.h>
#include <errno.h>

int main(int argc, char *argv[])
{
	int fd;
	int ret;

	if (argc<2) {
		fprintf(stderr, "Usage: %s file_name", argv[0]);
		fprintf(stderr, "Advises kernel that pages from file_name wont be needed soon.\n");
		exit(1);
	}
		
	fd = open(argv[1], O_RDWR);

	if (fd < 0) {
		perror("open");
		abort();
	}

	ret = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
	if (ret) {
		fprintf(stderr, "fadvise returned %d.  errno=%d", ret, errno);
		abort();
	}
	exit(0);
}
