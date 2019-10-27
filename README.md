# DeepJanus
replication package submitted to ICST2020. The repository contains three main elements: (1) the experimental data in xls format, (2) the folder DeepJanus-BeamNG and (3) the folder DeepJanus-MNIST

## BeamNG

Tested on a machine featuring an i9 processor, 32 GB of RAM, and an Nvidia 2080 TI GPU with 11GB of memory and a Windows OS. Python version 3.6.5 64-bit.

### Requirements

Most of the requirements have been collected in the requirements.txt file. Execute the following command

    pip install -r requirements.txt

Moreover, DeepJanus requires the installation of the BeamNG.research simulator. Please refer to

    https://beamng.gmbh/research/

### Running DeepJanus.BeamNG

To set up the configuration, modify the files core/config.py and self-driving/beamng_config.py.

To run DeepJanus, launch self-driving/main_beamng.py

## MNIST

Tested on a machine featuring an i7 processor, 16 GB of RAM, an Nvidia GeForce 940MX GPU with 2GB of memory and an Ubuntu OS.

### Requirements

Potrace on Python, please refer to

    https://pypi.org/project/pypotrace/

### Running DeepJanus.MNIST

To set up the configuration, modify the file properties.py.

To run DeepJanus, launch main.py
