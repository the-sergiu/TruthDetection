# Docker Image Build

### Prerequisites
- Docker and the implicit setup.

**1. Switch branch to "docker"**

Just run:
```
git checkout docker

```

**2. Just build the docker image**
```
docker build -t td .
```

**3. Run the docker image**
```
docker run -p 8888:8888 td
```
The docker image should run on the 8888 ports because that's the implicit port Jupyter Notebook usually runs on. That's also reflected in the Dockerfile.