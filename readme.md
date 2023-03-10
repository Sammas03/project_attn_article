### weather-rnn版本

v1  p*(c_t+temp_p==1) + (1-p)*wcode
在进入rnn解码前进行向量相加

v2 p*d_t + (1-p)*wcode
在进入fc_out层前加上wcode


v3 p*(c_t) + (1-p)wcode
在进入rnn解码前进行向量相加

v4 cat((d_t,wcode),dim=2)
在进入fc_out 前进行向量拼接

### talent 
添加cnn结构自动发现子序列

### statistic 里面存放了子序列图的统计代码

### 3090 ana-系列中存放了其余的图形代码