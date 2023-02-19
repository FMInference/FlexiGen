/*
 * userspace pagecache management using LD_PRELOAD, fadvise and
 * sync_file_range().
 *
 * John C. McCabe-Dansted <gmatht@gmail.com>
 * March, 2008
 *
 * Andrew Morton <akpm@linux-foundation.org>
 * March, 2007
 *
 * TODO: make this thread safe.
 *       make this more configurable on the commandline
 *
 */

#define _XOPEN_SOURCE 600
#include <fcntl.h>
#include <assert.h>
#include <time.h>


#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <limits.h>
#include <errno.h>

#include "sync_file_range.h"

#ifndef IGNORE_READS
#define COUNT_READS
#endif

enum fd_state {
	FDS_UNKNOWN = 0,	/* We know nothing about this fd */
	FDS_IGNORE =1,		/* Ignore this file (!S_ISREG?) */
	FDS_ACTIVE =2,		/* We're managing this file's pagecache */
	FDS_LAZY =3,		/* We were asked to close this file, but we are keeping it open */
};

struct fd_status {
	enum fd_state state;
	size_t bytes_written;
	time_t seconds;
};

#ifdef DEBUG
#define FDEBUGF fprintf
#else
#define FDEBUGF do_nothing
static int do_nothing() {
	return 0;
}
#endif

/*
 * These are set from the environment
 */
static unsigned long pagecache_max_bytes;
static unsigned long pagecache_chunk_size;
static unsigned long lazy_close_timeout;
static unsigned long wait_secs; /* assume older files already written */

static int pagesize;

/*
 * Global state
static unsigned long pagecache_size;
 */

static unsigned long pagecache_size_write;
#ifdef COUNT_READS
static unsigned long pagecache_size_read;
#endif 

#define LAZY_CLOSE_FDS

#ifdef LAZY_CLOSE_FDS
int max_lazy_fds=0;
static int *lazy=NULL;
static int lazy_first=0;
static int lazy_fds=0;


static void lazy_purge();

static int lazy_next(int x) {
	x++;
	if (x>=max_lazy_fds) {
		x=0;
	}
	return x;
}

static int lazy_i(int i) {
	int x=i+lazy_first;
	if (x>=max_lazy_fds) x -= max_lazy_fds;
	return x;
}

static int lazy_last() {
	return lazy_i(lazy_fds-1);
}
#endif

static struct fd_status *get_fd_status(int fd)
{
	static struct fd_status *fd_status;
	static int nr_fd_status;	/* Number at *fd_status */

	if (fd < 0)
		abort();
	if (fd + 1 > nr_fd_status) {
		fd_status = realloc(fd_status, sizeof(*fd_status) * (fd + 1));
		assert(fd_status!=NULL);
		memset(fd_status + nr_fd_status, 0,
			sizeof(*fd_status) * (fd + 1 - nr_fd_status));
		nr_fd_status = fd + 1;
		FDEBUGF(stderr,"Largest FD: %d\n",fd);
	}
	return &fd_status[fd]; /*FIXED BUG*/
}

static void grown_pagecache(int fd, size_t count, unsigned long *pagecache_size)
{
	if ((*pagecache_size) > pagecache_max_bytes) {
		off_t off = lseek(fd, 0, SEEK_CUR);
		if (off > pagesize)
			posix_fadvise(fd, 0, off - pagesize,
					POSIX_FADV_DONTNEED);
		(*pagecache_size) = 0;
	}
}


static void grown_pagecache_write(int fd, size_t count)
{
	FDEBUGF(stderr,"before grown_pagecache_write: %ld\n",pagecache_size_write);
	grown_pagecache(fd,count,&pagecache_size_write);
	FDEBUGF(stderr,"after grown_pagecache_write: %ld\n",pagecache_size_write);
}

#ifdef COUNT_READS
static void grown_pagecache_read(int fd, size_t count)
{
	grown_pagecache(fd,count,&pagecache_size_read);
}
#endif

/*
 * Work out if we're interested in this fd
 */
static void inspect_fd(int fd, struct fd_status *fds)
{
	struct stat stat_buf;

	if (fstat(fd, &stat_buf))
		abort();
	assert(fds->state==FDS_UNKNOWN);
	assert(fds->bytes_written==0);
	fds->bytes_written = 0;
	if (S_ISREG(stat_buf.st_mode) || S_ISBLK(stat_buf.st_mode))
		fds->state = FDS_ACTIVE;
	else
		fds->state = FDS_IGNORE;
}

static void write_was_called(int fd, size_t count)
{
	struct fd_status *fds;

#ifdef DUP_CLOSE
	assert(fd<200);
#endif

	pagecache_size_write += count;

	if (pagecache_max_bytes == 0)
		return;

#ifdef LAZY_CLOSE_FDS

	if (max_lazy_fds)
		lazy_purge();
#endif	

	fds = get_fd_status(fd);
	if (fds->state == FDS_UNKNOWN)
		inspect_fd(fd, fds);
	if (fds->state == FDS_IGNORE)
		return;
	grown_pagecache_write(fd, count);
	if (pagecache_size_write==0) {
	        fds->bytes_written = 0;
	} else {
		fds->bytes_written += count;
	}
}

#ifdef COUNT_READS
static void read_was_called(int fd, size_t count)
{
	struct fd_status *fds;

	if (pagecache_max_bytes == 0)
		return;

	fds = get_fd_status(fd);
	if (fds->state == FDS_UNKNOWN)
		inspect_fd(fd, fds);
	if (fds->state == FDS_IGNORE)
		return;
	pagecache_size_read+=count;
	grown_pagecache_read(fd, count);
}
#endif

static void close_was_called(int fd)
{
	struct fd_status *fds;

	if (pagecache_max_bytes == 0)
		return;

	fds = get_fd_status(fd);
	if (fds->bytes_written > 0 )  {
	if (time(NULL)-fds->seconds > wait_secs ) { 
		/* >=  /proc/sys/vm/dirty_writeback_centisecs */

		sync_file_range(fd, 0, LONG_MAX,
			SYNC_FILE_RANGE_WRITE|SYNC_FILE_RANGE_WAIT_AFTER);
		pagecache_size_write -= fds->bytes_written;
		fds->bytes_written=0;
#ifdef COUNT_READS
	}}
	posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#else
		posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
	}}
#endif
	fds->state = FDS_UNKNOWN;
}

static void parse_env(void)
{
	char *e;

	e = getenv("PAGECACHE_MAX_BYTES");
	if (e) {
		pagecache_max_bytes = strtoul(e, NULL, 10);
		pagecache_chunk_size = pagecache_max_bytes / 2;
		e = getenv("PAGECACHE_CHUNK_SIZE");
		if (e)
			pagecache_chunk_size = strtoul(e, NULL, 10);
	}
#ifdef LAZY_CLOSE_FDS
	e = getenv("PAGECACHE_MAX_LAZY_CLOSE_FDS");
	if (e) {
		max_lazy_fds=strtoul(e, NULL, 10);
		if (max_lazy_fds > 0) {
			lazy=malloc(max_lazy_fds*sizeof(int));
		}
	}
	e = getenv("PAGECACHE_LAZY_CLOSE_TIMEOUT");
	if (e) { 
		lazy_close_timeout=strtoul(e, NULL, 10); 
	} else {
		lazy_close_timeout=0; 
	}
		
#endif
	e = getenv("PAGECACHE_WRITEBACK_SECS");
	if (e) { 
		wait_secs=strtoul(e, NULL, 10); 
	} else { 
		wait_secs=20;
	}
}

/*
 * syscall interface
 */

static ssize_t (*_write)(int fd, const void *buf, size_t count);
static ssize_t (*_pwrite)(int fd, const void *buf, size_t count, off_t offset);
#ifdef COUNT_READS
static ssize_t (*_read)(int fd, void *buf, size_t count);
static ssize_t (*_pread)(int fd, void *buf, size_t count, off_t offset);
#endif
static int (*_close)(int fd);
static int (*_dup2)(int oldfd, int newfd);

static int symbols_loaded;

static void __load_symbols(void)
{
	void *handle;

	handle = dlopen("libc.so.6", RTLD_LAZY);
	if (!handle)
		abort();

	dlerror();
	_write = dlsym(handle, "write");
	if (dlerror())
		abort();

	_pwrite = dlsym(handle, "pwrite");
	if (dlerror())
		abort();

#ifdef COUNT_READS
	dlerror();
	_read = dlsym(handle, "read");
	if (dlerror())
		abort();

	_pread = dlsym(handle, "pread");
	if (dlerror())
		abort();
#endif
	_close = dlsym(handle, "close");
	if (dlerror())
		abort();

	_dup2 = dlsym(handle, "dup2");
	if (dlerror())
		abort();

	dlclose(handle);

	parse_env();

	pagesize = getpagesize();

	symbols_loaded = 1;
}

static inline void load_symbols(void)
{
	if (!symbols_loaded)
		__load_symbols();
}

ssize_t write(int fd, const void *buf, size_t count)
{
	assert(buf);
	load_symbols();
	write_was_called(fd, count);
	assert(buf!=NULL);
	return (*_write)(fd, buf, count);
}

ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset)
{
	load_symbols();
	write_was_called(fd, count);
	return (*_pwrite)(fd, buf, count, offset);
}

#ifdef COUNT_READS
ssize_t read(int fd, void *buf, size_t count)
{
	load_symbols();
	read_was_called(fd, count);
	return (*_read)(fd, buf, count);
}

ssize_t pread(int fd, void *buf, size_t count, off_t offset)
{
	load_symbols();
	read_was_called(fd, count);
	return (*_pread)(fd, buf, count, offset);
}
#endif

static int real_close(int fd) {
	FDEBUGF(stderr,"-- real close on: %d\n", fd);
	close_was_called(fd);
	return (*_close)(fd);
}

#ifdef LAZY_CLOSE_FDS
static void lazy_purge_first() {
	assert(lazy_fds>0);
	real_close(lazy[lazy_first]);
	lazy_first=lazy_next(lazy_first);
	lazy_fds--;
}

static void lazy_purge() {
	time_t t;
if (lazy_close_timeout) {
	t=time(NULL);
	while (( pagecache_size_write>pagecache_max_bytes || 
	( (  get_fd_status(lazy[lazy_first])->seconds-t >lazy_close_timeout  ) )) && lazy_fds>0) {
		FDEBUGF(stderr,"%ld > %ld #lazy_fds: %d\n",pagecache_size_write,pagecache_max_bytes,lazy_fds);
		lazy_purge_first();
	}
} else {
	while ( (pagecache_size_write>pagecache_max_bytes) && lazy_fds>0) {
		FDEBUGF(stderr,"%ld > %ld #lazy_fds: %d\n",pagecache_size_write,pagecache_max_bytes,lazy_fds);
		lazy_purge_first();
	}
}	
}


static int lazy_close(int fd) {
	struct fd_status *fds;
#ifdef DUP_CLOSE
	struct fd_status *newfds;
	int newfd;
#endif
	int ret_value=0;

	fds = get_fd_status(fd);
	
	FDEBUGF(stderr,"Lazy close on: %d, #lazy_fds: %d %ld\n", fd,lazy_fds,pagecache_size_write);
	if (fds->state==FDS_LAZY) {
		ret_value=EBADF;
		FDEBUGF(stderr,"EBADF on: %d\n", fd);
	} else {
		/*if (fds->state == FDS_UNKNOWN) {
			As far as we know, it hasn't been read or written to
			inspect_fd(fd, fds);*/
		if (fd < 3 || fds->bytes_written==0 || fds->state == FDS_IGNORE || fds->state == FDS_UNKNOWN)
		{
#ifdef DEBUG
			if (fds->state == FDS_IGNORE) FDEBUGF(stderr,"IGNORE ");
			if (fd<3) FDEBUGF(stderr,"<3 ");
			if (fds->bytes_written==0) FDEBUGF(stderr,"no_write ");
			FDEBUGF(stderr,"\n");
#endif
			ret_value=real_close(fd);
		} else {
			FDEBUGF(stderr,"-- Real Lazy Close\n");
			lazy_fds++;
			if (lazy_fds>max_lazy_fds) {
				FDEBUGF(stderr,"Too many lazy fds\n");
				lazy_purge_first();
			}
#ifdef DUP_CLOSE
			newfd=lazy_last()+200;
			dup2(fd,newfd);
			newfds = get_fd_status(newfd);
			(*newfds)=(*fds);
			ret_value =(*_close)(fd);
			lazy[lazy_last()]=newfd;
			fds = get_fd_status(fd);
			fds->state=FDS_UNKNOWN;
#else
			lazy[lazy_last()]=fd;
			fds->state=FDS_LAZY;
#endif
			fds->seconds=time(NULL);
		}
	}



	return ret_value;
}
#endif

int close(int fd)
{
	load_symbols();
	FDEBUGF(stderr,"close(%d)\n",fd);
#ifdef LAZY_CLOSE_FDS
	if (max_lazy_fds) 
		return lazy_close(fd);
	else
		return real_close(fd);
#else
	return real_close(fd);
#endif
}

int dup2(int oldfd, int newfd)
{
	load_symbols();
	FDEBUGF(stderr,"dup2(%d,%d)\n",oldfd,newfd);
	/* surely we want to close newfd instead? 
	 * close_was_called(oldfd); */ 
	close_was_called(newfd);
	return (*_dup2)(oldfd, newfd);
}
