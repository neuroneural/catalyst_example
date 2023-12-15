# catalyst_example
an example of the training file supporting distributed training and curriculum learning with Catalyst

# installation

1. Create and populate the environment
```
MYNEWENV="" # write a name of your environment in quotes, like "torch"
conda create --name ${MYNEWENV} python=3.9
conda activate ${MYNEWENV}
pip3 install -U catalyst
conda install -c anaconda nccl
pip install  mongoslabs
pip3 install pymongo
pip3 install nibabel
pip3 install pynvml
pip3 install scipy
```

2. Use the code in this repository to train your model
   1. The main file is `curriculum_training.py`
   2. Edit everything.
   3. Do not forget to set WANDBTEAM to the value of your team, so your logs are in the correct place
   4. Mainly make sure that `get_model` method of the `CustomRunner`
      class initializes your model

