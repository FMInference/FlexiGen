#!/bin/sh
#Allows page-cache management to keep 200 "closed" files open to improve performance, and does not attempt to limit the read cache
SO_NAME=`echo "$0" | sed s/-lazy200.sh$/-ignore-reads.so/g` 
if test ! -e "$SONAME"
then
	SO_NAME=`echo "$SONAME" | sed s/bin/lib/`
fi
if test -e $SO_NAME
then
	export LD_PRELOAD=$SO_NAME
	export PAGECACHE_MAX_BYTES=$((4096 * 512))
	export PAGECACHE_MAX_LAZY_CLOSE_FDS=200
	exec $*
else
	echo Could not open $SO_NAME
fi
