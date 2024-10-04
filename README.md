# nano-pgo
 ![example results](docs/results/visualization/readme.png)
- For an education purpose
- From-scratch pose-graph optimization implementation
- A single file
- Miminum dependencies: numpy/scipy/sksparse (and open3d for visualization).
    - numpy for basic matrix handling 
    - scipy for basic rotation functions and sparse matrix containers 
    - sksparse for cholmod and solve function
    - open3d for large-sized point cloud (pose-graph) visualization

## Preparation (Dependencies)
- Recommend to use python<3.12 and numpy<2, for example,
    - `$ python3.11 -m venv ~/envs/py311`
    - `$ source ~/envs/py311/bin/activate`
    - `$ pip install "numpy<2.0"`
    - `$ pip install scipy` 
    - `$ sudo apt-get install libsuitesparse-dev` 
    - `$ pip install scikit-sparse`
    - `$ pip install matplotlib`
    - `$ pip install open3d`

## How to use 
- `$ python nano_pgo.py`
- It's also recommended to compare the results from GTSAM (better and faster!) by using `baseline_gtsam.py`.
- Note that the nano_pgo.py's goal is a maximized transparency of all logics and theories from state representations to building and solving linear systems.

## Goal 
- Understand 
    - what is pose-graph optimization
    - what is a g2o-format pose-graph data 
    - what is se(3) and SE(3) (the tangent space and the manifold)
    - what is iterative least-square optimization and solving normal equation
    - what is the error and jacobians of between factors and how to be derived
    - why sparse solver is necessary (here, we used sksparse.cholmod)
    - why damping is necessary (i.e., LM iterative optimization method)
    - why robust loss is necessary (here, we used Cauchy deweighting)
    - how to use GTSAM APIs.
    - what is the real-world problems, use-cases, and state-of-the arts
    
## TODO
- Equipped with better initialization strategies (e.g., rotation averaging) 
- Use more theoretically accurate or automatically generated Jacobian (e.g., Symforce)
- Detailed teaching materials

## Acknowledgement 
- Datasets from https://lucacarlone.mit.edu/datasets/

