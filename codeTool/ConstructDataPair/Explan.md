#### 构造数据对<wa or tle, ac>
1.通过OrangizeRecord.py 处理生成每道题的记录到对应目录下的json文件中

#### 执行数据对中的代码填充"code1_test_status": [], 与"code1_test_score"的值
2. 执行的同时也应该验证数据是否合理，对于不合理数据进行剔除或标记
AddTestResultForRecord.py

project/
│
├── A/
│   ├── __init__.py
│   └── main.py
└── B/
    ├── __init__.py
    └── helpers.py


cd project
export PYTHONPATH=$(pwd)
python A/main.py
