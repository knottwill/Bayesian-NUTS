# S2 Coursework - William Knottenbelt, wdk24

<a href="#"><img src="https://img.shields.io/badge/python-v3.12.2-blue.svg?logo=python&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>

## Description

This project uses Bayesian inference to probe the position and source intensity of a lighthouse, based on lighthouse flash data in `lighthouse_flash_data.txt`.

Project structure:
```

```

## Usuage

To re-create the environment for the project you can either use conda or docker. Navigate to the root directory of the project and use one of the following:

```bash
# Option 1: re-create conda environment
$ conda env create -f environment.yml -n <env-name>

# Option 2: Generate docker image and run container
$ docker build -t <image_name> .
$ docker run -ti <image_name>
```

To re-produce all results/figures presented in the report, use:

```bash
$ python main.py --data ./lighthouse_flash_data.txt --output_dir ./plots
```

The plots will be saved to `--output_dir` and other results will be printed to the terminal.

### Timing

The main script took ~2 minutes to run on my personal laptop with the following specifications:
- Chip:	Apple M1 Pro
- Total Number of Cores: 8 (6 performance and 2 efficiency)
- Memory (RAM): 16 GB
- Operating System: macOS Sonoma v14.0
