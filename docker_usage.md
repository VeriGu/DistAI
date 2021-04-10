# Getting Started Instructions

- Install docker.

    - for Windows/Mac OS: https://www.docker.com/get-started 

    - for Linux: https://docs.docker.com/engine/install/

- Download image file

- Load docker image
    ```
    docker load -i the_address_to_the_image
    ```
  
- Run docker container
    ```
    docker run -it distinv /bin/bash
    ```
    
- Test basic functionality: in the docker container, run
    ```
    cd /dist-inv
    python main.py Ricart-Agrawala
    ```

# Detailed Instructions

## Table 3 & 4

### Dist-inv

- Enter ```/dist-inv```  in the docker container

- For each protocol, run 
    ```
    python main.py PROTOCOL
    ```

The proved protocol with correct invariants attached will be written to ```outputs/PROTOCOL/PROTOCOL_inv.ivy```. 
The script will print the counterexample number and invariant number. 
For the number of domains, variables, and maximum literal, one can directly inspect the output file. 
The script also prints the runtime breakdown among simulation/learning/refinement.

We found that the docker container causes a 15-60% runtime slowdown. To get the most accurate runtime numbers, consider the native installation guide in ```/dist-inv/install.txt```

### I4

- Enter ```/myI4```  in the docker container

- Enter Python 2 environment by 
    ```
    conda activate py2
    ```


- To reproduce the numbers in Table 3, for each protocol, run
    ```
    python myI4.py PROTOCOL -final|-total
    ```

- To run with a specific instance size, use mytest.sh, e.g.,
    ```
    bash ./mytest.sh chord “node=4”
    ```
  
- When finished, switch back to Python 3 by
    ```
    conda activate base
    ```



### FOL-IC3

- Enter ```/myfolic3``` in the docker container

- Run “forall” test for a specific protocol:
    ```
    python mypyvy/src/mypyvy fol-ic3 --logic=universal pyvs/PROTOCOL.pyv
    ```
  
- Run default test for a specific protocol:
    ```
    python mypyvy/src/mypyvy fol-ic3 --logic=fol pyvs/PROTOCOL.pyv
    ```
  
When running tests, lots of intermediate messages will show on screen, and the time will show when the test finishes.

The folic3 algorithm is randomized and the running time may vary. For protocols that require a long time to run, the real running time may range from about 10 min to more than 1 hour.
For some protocols, we also provide the version with automaton hints. Tests can be run using the file ```PROTOCOL_with_automaton.pyv```.



## Figure 6

In ```DistInv/```, run 
```
python tradeoff_figure.py
```

The figure will be written to ```tradeoff.png``` in the current directory.