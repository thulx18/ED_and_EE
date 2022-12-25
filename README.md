## NLP课程大作业：中文事件抽取任务

#### 1. 数据集

- **DuEE：** 

#### 2. 预训练模型

- **chinese-roberta-wwm-ext：** [huggingface/hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)


#### 3. 样例输出

$ python event_etraction.py

截至6月18日13时20分，四川省宜宾市长宁县6.0级地震已造成13人遇难。\\
Event 0 || type: 灾害/意外-地震 , trigger: 地震\\
    Argument:\\
        死亡人数 : 13人\\
        震级 : 6.0级\\