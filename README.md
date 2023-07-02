# InstructforNER
论文《InstructGPT在命名实体识别任务中的表现和挑战》代码

```
.
├── README.md
├── data
│   ├── pipe.py
│   └── prefix.py
├── few_shot.py # 运行few-shot的主文件
├── metrics.py
├── model
│   ├── cot_model.py
│   ├── icl_model.py
│   ├── utils.py # 需要在api_keys填入自己openai账号的api_key
│   └── zero_shot_model.py
└── zero_shot.py # 运行few-shot的主文件

2 directories, 10 files
```