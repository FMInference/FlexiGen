#!/bin/sh
if test -z "$1"
then
	echo Usage "$0 FILE ..."
	echo continuously tells the kernel that the pages in FILE will not be used again
	exit
fi

fadv=`echo $0 | sed s/.sh$//g`
file="$1"
while true
do
	#echo "$fadv" "$file"
	"$fadv" "$file"
	sleep 5
done
