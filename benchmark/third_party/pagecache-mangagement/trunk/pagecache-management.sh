#!/bin/bash
usage="Usage: $(basename $0): [-rd] [-lmktT value] -- command [args]"
rflag=
lflag=
size=30

assert_arg_nonneg() {
	if echo "$OPTARG" | grep -v "^[[:digit:]$1]"
	then
		echo -n "Option '$OPTION' needs non-negative numeric argument$2" >&2
		echo " but \"$OPTARG\" was provided" >&2
		echo "$usage" >&2
		exit 1
	fi
}


if test -e  /proc/sys/vm/dirty_writeback_centisecs
then
	centisecs=$(cat /proc/sys/vm/dirty_writeback_centisecs)
	export PAGECACHE_WRITEBACK_SECS=$(($centisecs/100+2))
	#echo $PAGECACHE_WRITEBACK_SECS
else
	echo Could not find /proc/sys/vm/dirty_writeback_centisecs, are we really using a Linux kernel?
fi

export SO_NAME=`echo "$0" | sed s/.sh$/.so/g | sed 's/^[[:alnum:]]/.\/&/g'` 
echo $SO_NAME
while getopts 'rl:k:m:t:T:d' OPTION
do
	case "$OPTION" in
	r)	rflag=1 # ignore reads
		;;
	l)	lflag=1 # max # of files to lazy close 
		lval="$OPTARG"
		export PAGECACHE_MAX_LAZY_CLOSE_FDS="$lval"
		assert_arg_nonneg 
		;;
	t) 	export PAGECACHE_WRITEBACK_SECS="$OPTARG"
		;;
	T) 	export PAGECACHE_LAZY_CLOSE_TIMEOUT="$OPTARG"
		echo $OPTARG
		assert_arg_nonneg = " or '='"
		;;
	k)	size="$OPTARG"
		assert_arg_nonneg
		;;
	m)	size=$(("$OPTARG"*1024))
		assert_arg_nonneg
		;;
	d)	DEBUG=1
		;;
	?)	echo "$usage" >&2
		exit 2
	nspect_fd	;;
	esac
done

if echo "$PAGECACHE_LAZY_CLOSE_TIMEOUT" | grep "^="
then
	PAGECACHE_LAZY_CLOSE_TIMEOUT="$PAGECACHE_WRITEBACK_SECS"
fi
echo $PAGECACHE_LAZY_CLOSE_TIMEOUT

	

shift $(($OPTIND - 1))

if test -z "$1"
then
	echo "$usage" >&2
	exit 2
fi

if test ! -e "$SO_NAME"
then
	SO_NAME=`echo "$SO_NAME" | sed s/bin/lib/`
fi

if test "$rflag"
then 
	SO_NAME=`echo "$SO_NAME" | sed s/.so$/-ignore-reads.so/`
fi
	
#if test "$lflag"
#then
#	export PAGECACHE_MAX_LAZY_CLOSE_FDS="$lval"
#fi


	
#export LD_PRELOAD=$(which $SO_NAME)
#if test -z $LD_PRELOAD
#then
	if test -e $SO_NAME
	then
		export LD_PRELOAD=$SO_NAME
		if test -z "$PAGECACHE_MAX_BYTES"
		then 
			export PAGECACHE_MAX_BYTES=$((1024 * $size))
		fi
		if test "$DEBUG"
		then
			gdb "$@"
		else 
			exec "$@"
		fi
		#gprof $*
	else
		echo Could not open $SO_NAME
	fi
#fi

