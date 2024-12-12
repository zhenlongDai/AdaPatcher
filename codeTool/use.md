### Create an docker instance for code execution

#### 1. Download docker image
Download address://github.com/criyle/go-judge?tab=readme-ov-file

#### 2. Create a docker instance
```
sudo docker run -it --privileged --shm-size=256m -p 5050:5050 --name=go-judge criyle/go-judge
```

#### 3. Install the Python environment

```
apt update && apt install python3
```

#### For performing operations inside a running Docker container (if needed)
```
docker exec -it go-judge /bin/bash
```
#### Restart (if necessary)
```
sudo docker restart go-judge 
```

### Instructions for use
#### A simple usage example
1.Modify codecontent and test points in the directory (test_directory) in codeTool.ExecutiveProgram.TestExample.RunProgramAndTestPostion.Py 

2.In the CodeTool/ directory, run the following command:
```
python -m ExecutiveProgram.TestExample.RunProgramAndTestPostion
```