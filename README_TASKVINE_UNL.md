# Execute materialize\_branches from coffea-casa dev and using the t3 at UNL 


## Conda environment with taskvine workers:

1. In a terminal, ssh to swan.unl.edu
1. From there, ssh to t3.unl.edu
1. Create a conda environment to get the taskvine workers executable:
	1. conda create -c conda-forge -n coffea-casa-min python=3.10.12 ndcctools
1. Optional, once: add basic coffea related packages to environment, not needed if using docker images at execution
	1. conda activate coffea-casa-min
	1. conda install -c conda-forge xrootd
	1. pip install fsspec-xrootd dask distributed coffea

## Execute notebook from coffea-casa-dev

1. In materialize\_branches.ipynb notebook, change `executor = "dask"` to `executor = "taskvine"`.
1. In a terminal, ssh to swan.unl.edu
1. From there, ssh to t3.unl.edu
1. Get a condor token: `mkdir -p ~/.condor/tokens.d && condor_token_fetch > ~/.condor/tokens.d/my_token`
1. Activate the environment: `conda activate coffea-casa-min`
1. Launch the workers:
```sh
casa_user_id=btovar-40nd-2eedu    # get user id from a terminal at coffea-casa: h=$(hostname); echo ${h#jupyter-}

casa_host=$casa_user_id.dask-worker.cmsaf-dev.flatiron.hollandhpc.org

docker=hub.opensciencegrid.org/coffea-casa/cc-dask-alma8:2024.04.05

number_of_workers=1

cores=12

memory=$((cores*3000))

disk=$((cores*6000))

vine_submit_workers -Tcondor -t300 --ssl -E"--transfer-port 1088" --cores $cores --memory $memory --disk $disk --docker-universe $docker $casa_host 8786 $number_of_workers
```
	
