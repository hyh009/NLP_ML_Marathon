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
★ 為什麼naïve？   →    假設所有的輸入特徵都是**彼此獨立**、且**不管順序**。<br>
★ scikit-learn API比較↓
| **scikit-learn API** |**Note** |
| :-------------: | :-----:|
| Naïve Bayes Multinomial|--- |
| Naïve Bayes Gaussian| 假設特徵成**高斯常態分布** | 
| Naïve Bayes Binary| 特徵必須為**二元分類** 0/1 |

**3. Decision Tree [作業-計算亂度&算法比較](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day24_%E6%B1%BA%E7%AD%96%E6%A8%B9%E4%BD%9C%E6%A5%AD.ipynb)**<br>
★ 適合處理有缺失值屬性的樣本。<br>
★ 容易overfitting。<br>
★ Decision Tree 算法比較↓<br>
| **算法名稱** |**分割準則** |
| :-------------: | :-----:|
| ID3| Entropy(Information Gain)  |
| C4.5| Entropy(Information Gain Ratio) | 
| CART|Gini Index|

**4. Random Forest [作業-實作隨機森林](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day27_%E5%AF%A6%E4%BD%9C%E6%A8%B9%E5%9E%8B%E6%A8%A1%E5%9E%8B_%E4%BD%9C%E6%A5%AD.ipynb)<br>**
★ Decision Tree 的集成學習(Ensemble learning) Bagging。<br>
★ 假設每個分類氣兼具有差異 & 每個分類自單獨表現夠好(Accuracy>0.5)<br>
★ 隨機抽選樣本和特徵，相較Decision Tree不易overfitting。<br>
★ 抗noise力較好(採隨機抽樣)，泛化性較佳。<br>
★ 對資料量小 & 特徵少的資料效果不佳。<br>

**5. Adaboost [作業-用樹型模型進行文章分類](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day28_%E5%AF%A6%E4%BD%9CTreeBase%E6%A8%A1%E5%9E%8B_%E4%BD%9C%E6%A5%AD%20.ipynb)<br>**
★ Decision Tree 的集成學習(Ensemble learning) Boosting。<br>
★ 將模型以序列的方式串接，透過過加強學習前一步的錯誤，來增強這一步模型的表現。<br>
★ 對noise敏感。<br>
★ 訓練時間較長。<br>
##### ※Random Forest & Adaboost 比較<br>
| **Random Forest** |**Adaboost** |
| :-----: | :-----:|
| 每個Decision Tree都是Full sized，不會事先決定最大分支數| 每個Decision Tree通常只有一個節點&2片葉(stump)|
| 會使用多個variables進行決策| 一次只用一個variables | 
| 最終決策時，採均勻多數決|最終決策時，每棵樹票不一樣大|
| 樹的順序無影響|樹間有關連(序列架構)，順序很重要|
| 泛化能力佳 抗noise能力較高| 對noise敏感 |
| 使用均勻採樣bootstrap(抽中放回)|上個模型分錯的樣本，下次被抽中的機率較高|
##### ※樹型(Tree base)模型衡量指標<br>
**衡量訊息亂度&分割準則**<br>
* ##### 熵(Entropy) & Gini 不純度（Gini Impurity = Gini Index)<br>
  ◎衡量一個序列中的混亂程度，值越高越混亂   →   最終目的是最終leaves的亂度最小化。<br>
  ◎數值都在 0 ~ 1之間，0 代表皆為同樣的值(類別)。<br>
* ##### Informarion Gain (IG)<br>
  ◎計算方式：<br>
  1. 先各計算node和分支後的左leaf、右leaf的訊息亂度 & 左右leaf的sample數比例。<br>
   →  Ex: 右leaf的sample比例 = 右邊sample數/全部sample數。<br>
  2. IG = node訊息亂度 - sample比例(右) X 右leaf訊息亂度 - sample比例(左) X 左leaf訊息亂度。<br>

* #### 交叉驗證(cross validation)<br>
**1. KFold<br>**

* #### Bias-Variance Tradeoff [作業-了解何謂Bias-Variance Tradeoff](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day25_Random_forest%E4%BD%9C%E6%A5%AD.ipynb)<br>
  一般模型誤差可拆為三部分   →   **整體誤差(Total error)= 偏差(Bias) + 變異(variance) + 隨機誤差(random error => noise)**<br>
  Bias： 訓練集的預測結果和實際答案的落差。<br>
  Variance： 模型在測試資料集的表現 → 模型的泛化性指標。<br>
  **Bias過大 → underfitting** ， 但 **Bias小 Variance過大 → overfitting**。
  Bias-Variance Tradeoff就是在Bias 和 Variance 中取得一個最佳的平衡，目的是使**整體誤差下降**。
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
