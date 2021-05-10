# NLP_ML_Marathon
## NLP經典機器學習馬拉松 6大學習里程碑◢

[① Python NLP 程式基礎](#A)<br>

[② 詞彙與分詞技術](#B)<br>

[③ NLP 詞性標註](#C)<br>

[④ 經典詞彙向量化(詞袋分析/ TF-IDF / SVD&共現矩陣)](#D)<br>

[⑤ NLP 經典機器學習模型](#E)<br>

[⑥ 期末實務專題](#F)<br>


## 重點整理

### <a name="A">① Python NLP 程式基礎</a><br>
* #### 文字處理函數<br>
* #### 正規表達式<br>
★測試pattern可[至此](https://regex101.com/)<br>

### <a name="B">② 詞彙與分詞技術</a><br>
* #### 中文斷詞<br>
**1. Ckip<br>**
**2. jieba<br>**
* #### N-Gram<br>
馬可夫假設<br>

### <a name="C">③ NLP 詞性標註</a><br>
* #### jieba<br>

### <a name="D">④ 經典詞彙向量化(詞袋分析/ TF-IDF / SVD&共現矩陣)</a><br>
* #### Bag-of-words<br>
* #### TF-IDF<br>
* #### 共現矩陣→PPMI→SVD<br>
* #### 詞幹/詞條提取：Stemming and Lemmatization(英文)<br>
### <a name="E">⑤ NLP 經典機器學習模型(scikit-learn)</a><br>
* #### 機器學習模型</font><br>
**1. KNN<br>**
★ 適合資料少、維度小的資料。<br>
★ k值過大易underfitting;k值過小易overfitting。<br>
★ k可以先設定成k\*\*0.5再進行調整，且應盡量設成奇數(防止有票數相同無法判別類別的問題)。<br>

**2. Naïve Bayes<br>**
★ 為什麼naïve？   → 假設所有的輸入特徵都是**彼此獨立**、且**不管順序**。<br>
★ scikit-learn API比較↓
| **scikit-learn API** |**Note** |
| :-------------: | :-----:|
| Naïve Bayes Multinomial|--- |
| Naïve Bayes Gaussian| 假設特徵成**高斯常態分布** | 
| Naïve Bayes Binary| 特徵必須為**二元分類** 0/1 |

**3. Decision Tree<br>**
**4. Random Forest<br>**
**5. Adaboost<br>**
* #### 交叉驗證(cross validation)<br>
**1. KFold<br>**
* #### 延伸：其他Ensemble learning<br>
**1. Stacking<br>**
**2. Blending<br>**
### <a name="F">⑥ 期末實務專題</a><br>

* #### 自製中文選字系統<br>
* #### 建置新聞分類器<br>
* #### 垃圾郵件偵測器<br>
* #### 情緒分析<br>
* #### 潛在語意分析<br>
* #### 自動文件修改器(Trigram運用)<br>
* #### 聊天機器人(單輪對話)<br>
* #### Line Bot 聊天機器人(多輪對話)<br>
