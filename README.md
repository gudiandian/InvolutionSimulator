# InvolutionSimulator
《社会与市场中的计算问题选讲》课程大作业，“内卷”现象模拟。

## Software Dependencies

- networkx == 2.5.1
- sympy == 1.8
- matplotlib >= 3.4.1

## Run Simulation

```bash
python3 involution.py --k <involution level> --n <number of nodes> --e <number of edges>
```
`k`为一个大于0的可调的参数，k越大，节点越容易在别人比自己努力的时候受到别人影响，从而加剧内卷程度
`n`为网络中节点个数
`e`为网络中的边数