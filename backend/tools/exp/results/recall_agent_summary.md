### Recall Agent 消融：keyword-noLLM vs text vs image vs fused (N=30, K=30，相关性=同视觉品类)

| metric | none | text_only | image_only | fused | delta(T-N) | sig(T-N) | delta(I-N) | sig(I-N) | delta(F-N) | sig(F-N) | delta(F-T) | sig(F-T) | delta(F-I) | sig(F-I) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| precision@30 | 0.1633 | 0.1644 | 0.2356 | 0.2256 | 0.0011 | ns | 0.0722 | ** | 0.0622 | ** | 0.0611 | *** | -0.0100 | ns |
| ndcg@30 | 0.6001 | 0.5899 | 0.7559 | 0.7138 | -0.0102 | ns | 0.1559 | *** | 0.1137 | ** | 0.1239 | *** | -0.0422 | ns |
| recall@30 | 0.4365 | 0.4951 | 0.6867 | 0.6481 | 0.0586 | ns | 0.2503 | *** | 0.2116 | ** | 0.1530 | *** | -0.0387 | ns |
