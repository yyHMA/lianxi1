# Tacotron + MelGAN chinese

本项目任务是希望得到一个较好clone声音

本项目代码来自 https://github.com/chldkato/Tacotron-MelGAN-Korean

本项目主要分为4部分：1）Tacotron的base模型；2）MelGAN的base模型；3）Tacotron的微调模型；2）MelGAN的微调模

### Data

1. **base模型训练数据：标贝**

    * [biaobei](https://www.data-baker.com/open_source.html)

2. **数据目录**

   ```
   BZNSYP
     |- PhoneLabeling
         |- 000001.interval
         |- 000002.interval
     |- ProsodyLabeling
         |- 000001-010000.txt
     |- Wave
         |- 000001.wav
         |- 000002.wav
         |- 000003.wav
         |- 000004.wav
     |- metadata.csv（自己处理）
         000001|ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
         000002|jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
   ```
   
### Tacotron Training
1. **Preprocess**
   ```
   python tacotron-pre.py
   ```
     * tacotron-pre.py 根据BZNSYP/metadata.csv的文件名，获取音频路径，将wav转成mel数据，即kss-mel-xxx.npy
     * 运行后，将在tacotron-train文件夹中创建训练数据(.npy和train.txt)

2. **Train**
   ```
   python tacotron-train.py
   ```
     * logs-tacotron 保存模型和对齐图 （utils中，改成中文的处理方式，改了cleaners.py、symbols.py、numbers.py、text.py（），改了utils/text.py的text_to_sequence方法，删掉了cleaner_names参数等。plot.py修改了7、8、17行。）
     * 训练到 849k-step

3. **Inference**
    ```
    python tacotron-eval.py
    ```
    * tacotron-eval.py里编辑好要生成mel的拼音短句，tacotron-output保存推理的mel数据
 

### Melgan Training
1. **Preprocess**
   ```
   python melgan-pre.py
   ```
     * 将wav文件转成.mel文件,会在melgan文件夹下建立四个文件夹data，test，train，valid
     * train文件夹的数据需手动移动进去（包含.mel和.wav数据）
     * 10000个文件，2个测试集，2个验证集，9996个个训练集
     
2. **Train**
   ```
   python melgan-train.py
   ```
     * logs保存文件和valid音频
     * 训练到 5455k-step
     
3. **Inference**
    ```
    python melgan-eval.py
    ```
    * 将tacotron-output的mel数据合成wav数据，保存在melgan-output中

### tacotron-elgan Inference

   ```
   python tacotron-melgan-interface.py
   ```     
   * tacotron和melgan和在一起的接口，可以在cpu上运行
   

### Clone voice
重复上述Tacotron Training和Melgan Training的步骤

数据：单个人约600+数据，数据格式同biaobei的数据

### experience
1. 微调的tacotron部分，需要较多步骤的迭代，才能有比较稳定的效果；melgan部分只需要少许的迭代训练

2. 微调出来的tacotron部分不稳定，有不出声音或长时间的尾音问题

### 样例
试听链接 http://htmlpreview.github.io/?https://github.com/yyHMA/lianxi1/blob/master/web.html
