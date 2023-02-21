/*
 * Stupid test for sync_file_range().  Do:
 *
 * sync
 * dd if=/dev/zero of=foo bs=1M count=100
 * sfr foo
 * grep Dirty /proc/meminfo
 *
 * and be sure that there's no dirty memory left
 */

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <limits.h>

#include "sync_file_range.h"

int main(int argc, char *argv[])
{
	int fd = open(argv[1], O_RDWR);
	int ret;

	if (fd < 0) {
		perror("open");
		abort();
	}

	ret = sync_file_range(fd, 0, LONG_MAX,
			SYNC_FILE_RANGE_WAIT_BEFORE|SYNC_FILE_RANGE_WRITE|
			SYNC_FILE_RANGE_WAIT_AFTER);
	if (ret) {
		perror("sync_file_range");
		abort();
	}

	return 0;
}
