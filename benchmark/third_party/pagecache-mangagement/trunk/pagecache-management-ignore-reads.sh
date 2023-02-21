#!/bin/sh
SO_NAME=`echo "$0" | sed s/.sh$/.so/g` 
if test ! -e "$SONAME"
then
	SO_NAME=`echo "$SONAME" | sed s/bin/lib/`
fi

if test -e $SO_NAME
then
	export LD_PRELOAD=$SO_NAME
	export PAGECACHE_MAX_BYTES=$((4096 * 512))
	exec $*
else
	echo Could not open $SO_NAME
fi

