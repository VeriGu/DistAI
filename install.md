# DistInv Installation Guide

This guide provides instructions for building DistInv from source. 

1. Download and install Anaconda from https://www.anaconda.com/products/individual. Use the latest version (Python 3.8).

2. The Ivy verification tool only works on Python 2. Install it by 
    ```
    $ conda create --name py2 python=2.7
    $ conda activate py2
    $ pip install ms-ivy
    ```
   
3. Configure Ivy path

    - Run `which ivy_check` to get the absolute path of the Ivy checker. We assume it is `ANACODNA_PATH/envs/py2/bin/ivy_check`.

    - Append the following line to `~/.bashrc`
        ```
        alias ivy_check="ANACODNA_PATH/envs/py2/bin/ivy_check"
        ```
    
    - Copy and replace the absolute path at `#define IVY_CHECK_PATH` in `src-c/InvRefiner.h`. (This is a workaround for calling Python2 from a Python3 conda environment. We would appreciate any suggestion to make this more elegant)

4. Install Python libraries 
    ```
    $ conda activate base
    $ conda install numpy scipy pandas
    ```

5. Build C++ source files
    ```
    $ cd src-c
    $ make
    $ source ~/.bashrc
    ```


## Notes

- We compared DistInv with I4 and FOL-IC3 in the paper. For native installation of I4, see https://github.com/GLaDOS-Michigan/I4. For FOL-IC3, see https://dl.acm.org/do/10.1145/3395650/abs/.

- If you want native installation of DistInv and I4 on the same machine, you should install Ivy as above and skip the Ivy installation step in I4. This ensures that Ivy is recognizable by anaconda.

- To get I4 working, besides the instructions provided by the authors, we needed to edit a line in file ``ANACONDA_PATH/envs/py2/bin/ivy_check``. We changed `from ivy.ivy_check import main` to `from ivy_check import main`.