# DistAI: Data-Driven Automated Invariant Learning for Distributed Protocols

DistAI is a tool to automatically infer inductive invariants to prove the correctness of distributed protocols. It takes a simulation-learning-refinement workflow. 
Given a protocol, DistAI first repeatedly simulate the protocol using randomized instances. This gives a collection of simulation traces (i.e., samples). 
Then DistAI applies an enumeration-based learning procedure to find invariants that hold true on all the samples. Finally, DistAI uses Ivy, a theorem prover built on top of Z3 SMT solver,
to refine the invariants using counterexamples until validated.

## Installation

You can build DistAI from source using the installation guide ```install.txt```. Native installation gives the most accurate runtime numbers.

Alternatively, one can use our [docker image](https://drive.google.com/file/d/1ogBU9KvZsvSRhXerY9Bv-MuiW9oOezBU/view?usp=sharing) with the guide ```docker_usage.md```. This image also includes I4 and FOL-IC3, the two systems DistAI compared with in the evaluation.

## Usage

#### Top-level API

Given the name of a distributed protocol, ```python main.py PROTOCOL``` simulates the protocol, learn the invariants and refine them. The proved protocol with correct invariants will be written to ```outputs/PROTOCOL/PROTOCOL_inv.ivy```

```
python main.py two_phase_commit
```

#### Step-by-step instructions

In ```src-py/```, we can use ```python translate.py PROTOCOL``` to generate the simulation script in Python from the Ivy source.
It accepts a conditional argument ```--min_size``` for the minimum instance size (i.e., initial template size). If omitted, DistAI will infer the minimum size from the protocol.

```
python translate.py distributed_lock --min_size="node=2 epoch=2"
```

The simulation script is written to ```auto_samplers/```. From there, run ```python PROTOCOL.py``` to simulate the protocol.

```
python database_chain_replication.py
```

The samples are written to ```traces/```, and a configuration file is generated in ```configs/```. Then, in ```src-c/```,
run ```./main PROTOCOL [MAX_RETRY]``` to learn and infer the invariants. ```MAX_RETRY``` is a conditional argument indicating how many times the solver should retry (i.e., incrementing max-literal) before giving up.

```./main blockchain 1```

The proved protocol with correct inductive invariants attached is written to ```outputs/PROTOCOL/PROTOCOL_inv.ivy```. From there, one can use
```ivy_check PROTOCOL_inv.ivy``` to validate its correctness.


#### Running new protocols

To use DistAI on a new distributed protocol, simply add the Ivy file at ```protocols/NEW_PROTOCOL/NEW_PROTOCOL.ivy```, and make an empty directory ```outputs/NEW_PROTOCOL/```. Then run ```python main.py NEW_PROTOCOL``` 

#### Reproduce Figure 6

Run ```python tradeoff_figure.py```. The output figure is saved at ```tradeoff.png```

## Structure

- protocols/:
  The 14 distributed protocols in Ivy. The mapping between protocol names in the paper and acronyms in this repository is given below.
  
  | protocol name           | acronym                    |
  |-------------------------|----------------------------|
  | async lock server       | multi_lock_server          |  
  | chord ring maintenance  | chord                      |
  | database replication    | database_chain_replication |
  | decentralized lock      | decentralized_lock         |
  | distributed lock        | distributed_lock           |
  | hashed sharding         | shard                      |
  | leader election         | leader                     |
  | learning switch         | switch                     |
  | lock server             | lock_server                |
  | Paxos                   | paxos                      |
  | permissioned blockchain | blockchain                 |
  | Ricart-Agrawala         | Ricart-Agrawala            |
  | simple consensus        | consensus                  |
  | two-phase commit        | two_phase_commit           |
  
- src-py/:
  The python source codes for protocol simulation
  - translate.py: parse an Ivy file and emits a Python simulation script
  - translate_helper.py: provides functionality for translate.py
  - ivy_parser.py: parse an Ivy expression and generates a syntax tree, used by translate.py
  
- src-c/:
  The C++ source codes for invariant enumeration and refinement
  - main.cpp: the top-level handler
  - basics.h/cpp: define types and data structures (e.g., how an invariant is represented)
  - preprocessing.h/cpp: reads and process the trace file and configuration file
  - Solver.h/cpp: the core of DistAI. Implements template projection, invariant enumeration, and invariant projection
  - InvRefiner.h/cpp: interact with Ivy to refine the invariants using counterexamples
  - Helper.h/cpp: provides functionality for other files
