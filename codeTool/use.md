### 创建实例

下载地址：https://github.com/criyle/go-judge?tab=readme-ov-file

#### 创建一个docker实例，终止后自动删除实例 
'''
sudo docker run -it --rm --privileged --shm-size=256m -p 5050:5050 --name=go-judge criyle/go-judge
'''

'''
sudo docker run -it --privileged --shm-size=256m -p 5050:5050 --name=go-judge criyle/go-judge
'''

#### 用于在运行中的 Docker 容器内执行操作
'''
docker exec -it go-judge /bin/bash
'''
#### 重启 

'''
sudo docker restart go-judge 
'''

### 使用说明
1.修改RunProgramAndTestPostion.py中的content与测试点所在目录

2.在CodeTool/目录下,执行以下命令：
'''
python -m ExecutiveProgram.TestExample.RunProgramAndTestPostion
'''