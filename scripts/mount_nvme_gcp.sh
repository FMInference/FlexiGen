# Mount NVMe drive on GCP instances.
#
# Usage:
# sudo bash mount_nvme_gcp.sh ~/
# sudo bash mount_nvme_gcp.sh /home/sqy1415/

if [ "$1" = "" ]
then
  echo "Usage: $0 mount_path"
  exit
fi

pvcreate /dev/nvme0n1 /dev/nvme0n2 /dev/nvme0n3 /dev/nvme0n4
pvs
vgcreate striped_vol_group /dev/nvme0n1 /dev/nvme0n2 /dev/nvme0n3 /dev/nvme0n4
vgs
lvcreate -i 4 -I 1m -l 100%VG -nstriped_logical_volume striped_vol_group
lvs
mkdir -p $1flexgen_offload_dir
mkfs -t xfs -f /dev/striped_vol_group/striped_logical_volume
mount /dev/striped_vol_group/striped_logical_volume $1flexgen_offload_dir
chmod a+rw $1flexgen_offload_dir
