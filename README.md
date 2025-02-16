# catalyst_example
an example of the training file supporting distributed training and curriculum learning with Catalyst

# installation

1. Create and populate the environment
```
MYNEWENV="" # write a name of your environment in quotes, like "torch"
conda create --name ${MYNEWENV} python=3.9
conda activate ${MYNEWENV}
pip install -r requirements.txt
```

2. Use the code in this repository to train your model
   1. The main file is `curriculum_training.py`
   2. Edit everything.
   3. Do not forget to set WANDBTEAM to the value of your team, so your logs are in the correct place
   4. Mainly make sure that `get_model` method of the `CustomRunner`
      class initializes your model

# How to run the code on Slurm

1. Edit the `submit-job.sh` file as follows:
   1. Set the environment variable `MYNEWENV` to the name of the environment you created in the #installation section.
   2. Set the job name to something meaningful, e.g., `curriculum_training`
   3. Set your email in the `#SBATCH --mail-user` line
   4. Set your partition in the `#SBATCH -p` line
2. Submit the job using the following command:
```
sbatch run_training.sh
```


# Gotchas to look out for:
1. Failure to connect to mongoDB
> Solution: change the MONGOHOST from "10.245.12.58" to "arctrdcn018.rs.gsu.edu" in `conf/vanilla_3class_gn_11chan32.16.1_exp01.yaml`

1. Failure to connect to wandb
> Solution: Ensure you have a wandb account, get a token and use it to login to wandb using `wandb login`. Also ensure that you change the WANDBTEAM and project from "hydra-meshnet" to "your-team-name" and "your-project-name" respectively in `conf/vanilla_3class_gn_11chan32.16.1_exp01.yaml`
