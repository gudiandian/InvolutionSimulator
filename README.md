# InvolutionSimulator
《社会与市场中的计算问题选讲》课程大作业，“内卷”现象模拟。

## Software Dependencies

- networkx == 2.5.1
- sympy == 1.8
- matplotlib >= 3.4.1

## Run Simulation

通过网络级联模型模拟“内卷”的网络效应。

```bash
python3 involution.py --n <number of nodes> --e <number of edges> -s <graph swap rate>
```

`n`为网络中节点个数

`e`为网络中的边数

## Run Analysis
探究平均卷度与图连通性（边的数量）的关系
```bash
python3 analysis.py --n <number of nodes> --e <number of edges>
```
