#!/bin/sh
if test -z "$2"
then
	echo Usage "$0 FILE COMMAND OPT1 OPT2 ..."
	echo Runs COMMAND while continuously telling the kernel that the pages in FILE will not be used again
else

	fadv_sh=`echo $0 | sed s/_command//g`
	binary=`echo $0 | sed s/.sh$//g`
	file=$1
	command=$2
	shift
	shift
	"$fadv_sh" "$file" &
	fadv_sh_pid=$!
	"$command" "$@"
	kill $fadv_sh_pid
fi
