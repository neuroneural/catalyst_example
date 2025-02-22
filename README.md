# catalyst_example
an example of the training file supporting distributed training and curriculum learning with Catalyst

# Installation

1. Create and populate the environment
```bash
MYNEWENV="" # write a name of your environment in quotes, like "torch"
conda create --name ${MYNEWENV} python=3.9
conda activate ${MYNEWENV}
pip install -r requirements.txt
```

2. Training Setup
   1. The main training script is `curriculum_training.py`
   2.  Mainly make sure that `get_model` method of the `CustomRunner` class initializes your model
   3. Create or modify the config file (e.g. `conf/vanilla_3class_gn_11chan32.16.1_exp01.yaml`) as follows:
      - Set `wandb.team` to your team name for proper logging

# How to run the code on Slurm

1. Configure `submit-job.sh` by changing the following:
   ```bash
   # Required changes:
   MYNEWENV=""        # Set to your conda environment name
   #SBATCH --job-name # Set meaningful job name
   #SBATCH --mail-user=your.email@domain.com
   #SBATCH -p your_partition
   #SBATCH -A your_account
   CONFIG="your_config"
   CONFIG_PATH="path/to/config"
   ```

2. Submit the job:
```bash
sbatch submit-job.sh
```

# Troubleshooting

1. Wandb Connection Issues
   - Create a wandb account and obtain an API token
   - Run `wandb login` with your token
   - Update `wandb.team` and `wandb.project` in your config file

2. MongoDB Connection
   - Development node: Uses "10.245.12.58"
   - Slurm nodes: Uses "arctrdcn018.rs.gsu.edu"
   - Connection is automatically handled based on `SLURM_JOB_ID` environment variable but you can also switch things up under the `mongo` section of the config file when you run into issues.
