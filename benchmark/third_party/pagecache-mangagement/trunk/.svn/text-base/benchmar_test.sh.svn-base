#export LD_PRELOAD=$(which $SO_NAME)
export LD_PRELOAD=./pagecache-management-lazy200.so
export PAGECACHE_MAX_BYTES=$((4096 * 256))
FROM=tmp/squashfs
TO=/media/sdb7/images/pagemanagement_for_ubiquity_test.dir
#gdb  cp -r $FROM $TO
valgrind  cp -r $FROM $TO
