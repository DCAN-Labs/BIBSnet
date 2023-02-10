## Dependencies 

docker or singularity (depending on what containerization software you will use for execution)

<br />

## Installation

We strongly recommend that users interested in producing segmentations run the containerized version of `CABINET`. However, developers may sometimes benefit from running the application outside of the container for testing. 

<br />

### Container

Container hosted here: https://hub.docker.com/r/dcanumn/cabinet

#### Singularity

    singularity pull cabinet.sif docker://dcanumn/cabinet

#### Docker

    docker pull dcanumn/cabinet:t1-only_t2-only
