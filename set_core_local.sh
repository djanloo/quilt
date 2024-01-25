# Compile with -ggdb flag
# Use like gdb path/to/my/executable path/to/core
# Inside gdb: bt full
ulimit -c unlimited
sudo systemctl stop apport.service
sudo systemctl disable apport.service
cat /proc/sys/kernel/core_pattern
