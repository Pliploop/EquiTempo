# EquiTempo
Reproduction of the paper : Equivariant Self-Supervision for Musical Tempo Estimation (ISMIR 2022)

## Contributing [Temp until we get the project off the ground]
- clone from branch:dev
- create local branch and set remote to same branch name
- when done, pull request to dev. Keep working on same local branch after pulling or create new branch, up to you


### Config shenanigans

A quick explanation on how the config system works. Each config is a python file which with each experiment log will be saved to a yaml file (to be later loaded for reproducibility and model architecture) via the GlobalConfig class.

- GlobalConfig is instanciated at the time of training -> imports all current configs and saves them to a yaml file in the form of a dict.
- When instanciated, all classes can be provided with a globalConfig object (loaded from yaml - class method) to overwrite configs in the config folder - if not, config folder classes are used.
- This way the at the start of each evaluation / test / training, a simple loading of a globalConfig file provides all the needed configs and the parameters trickle down through the classes.