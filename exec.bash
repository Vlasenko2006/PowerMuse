#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=muse
#SBATCH --nodes=1 # unfortunately 3 is the max on strand at the moment. 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=71:59:00
#SBATCH --account=ksm
#SBATCH --partition=pGPU
#SBATCH --error=e-muse_trans3.out
#SBATCH -o muse_trans3.out

#SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                    # Request all memory available on all nodes


#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/home/vlasenko/miniconda3/lib/



srun /gpfs/work/vlasenko/07/NN/fatenv/gpt2_finetuning_env/bin/python3.10 model_trans.py #download_and_print_data.py #resave_anomalies.py #SSTT2MDataset.py #remove_clim_sec.py #remove_climatology_redres_t2m.py #Lang5_self_long.py
