## Dependencies 

docker or singularity (depending on what containerization software you will use for execution)

<br />

## Installation

We strongly recommend that users interested in producing segmentations run the containerized version of `BIBSnet`. However, developers may sometimes benefit from running the application outside of the container for testing. 

<br />

### Container

Container hosted here: https://hub.docker.com/r/dcanumn/bibsnet

#### Singularity

    singularity pull bibsnet.sif docker://dcanumn/bibsnet:latest

#### Docker

    docker pull dcanumn/bibsnet:latest
