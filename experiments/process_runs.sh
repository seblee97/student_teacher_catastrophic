# Args:
# 1: whether or not to first scp folder from VM (0 = False, 1 = True)
# 2: datestamp of experiment on VM
# 3: folder name in cluster_results to which to save files
# 4: extra arguments to provide post process call

if [ $1 == "1" ]
then
    scp -r rfl568@rescomp1.well.ox.ac.uk:/well/saxe/users/rfl568/repos/student_teacher_cont/experiments/results/$2 \
    /Users/sebastianlee/Dropbox/Documents/Research/Projects/catastrophic/experiments/results/cluster_results/$3
else
    echo "scp completed as per first argument"
fi

for OUTPUT in $(ls /Users/sebastianlee/Dropbox/Documents/Research/Projects/catastrophic/experiments/results/cluster_results/$3)
do
    python main.py --ppp results/cluster_results/$3/$OUTPUT --ipf --nl $4
    python main.py --ppp results/cluster_results/$3/$OUTPUT $4
done
