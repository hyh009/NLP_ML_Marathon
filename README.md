# NLP_ML_Marathon
## NLP經典機器學習馬拉松 6大學習里程碑◢

[① Python NLP 程式基礎](#A)<br>

[② 詞彙與分詞技術](#B)<br>

[③ NLP 詞性標註](#C)<br>

[④ 經典詞彙向量化(詞袋分析/ TF-IDF / SVD&共現矩陣)](#D)<br>

[⑤ NLP 經典機器學習模型](#E)<br>

[⑥ 期末實務專題](#F)<br>


## 學習重點整理

### <a name="A">① Python NLP 程式基礎</a><br>
* #### 文字處理函數<br>
* #### 正規表達式<br>
★測試pattern可至[Regex101](https://regex101.com/)<br>

### <a name="B">② 詞彙與分詞技術</a><br>
* #### 斷詞的方法<br>
  ★
  
* #### 中文斷詞套件<br>
**1. CkipTagger [作業-使用CkipTagger進行各項的斷詞操作](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day7_CkipTagger%E4%BD%9C%E6%A5%AD_checkpoint.ipynb)<br>**<br>
  ★中研院開發，繁中斷詞能力較好，但速度較慢。<br>
  ★功能：WS：斷詞 / POS：詞性標注 / NER：實體辨識<br>
  ★需先下載預訓練權重<br>
  >from ckiptagger import data_utils, WS<br>
   data_utils.download_data_gdown("./")<br>
   
**2. jieba [作業-使用Jieba進行各項的斷詞操作](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day6_Jieba_%E4%BD%9C%E6%A5%AD%20.ipynb)**<br>

  ★繁中斷詞要先設定繁中字典。 -> jieba.set_dictionary('./dict.txt.big')<br>
  ★速度較CkipTagger快。<br>
  ★分為精確模式(default)、全模式(斷出全部可能性)、搜尋模式、Paddle模式<br>

* #### N-Gram [作業-N_Gram實作作業](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day9_%E5%9F%BA%E7%A4%8E%E8%AA%9E%E8%A8%80%E6%A8%A1%E5%9E%8BN_Gram%E5%AF%A6%E4%BD%9C%E4%BD%9C%E6%A5%AD.ipynb)<br>
  將文本從頭開始每N個字取一次詞，取得 len(文本)-N+1 長度的序列(當中每個物件為N個字詞)，並計算條件機率。<br>
  **馬可夫假設： 從目前狀態轉移 s 到下一個狀態 s' 的機率由 P(s'|s) 來決定 (在 s 的前提下 s’ 發生的機率)**<br>
  
  **原始算法↓**<br>
   > W=(W1W2W3...Wm)<br>
  P(W1W2W3...Wm) = P(W1) X P(W2|W1) X P(W3|W1,W2) X...X P(Wm|W1,W2...Wm-1) <br>
  
  **引入馬可夫假設(僅考慮前一個狀態轉移到下一個狀態的機率)↓**<br>
   > W=(W1W2W3...Wm)<br>
  P(Wm|W1,W2...Wm-1) = P(Wm|Wm-n+1,Wm-n+2...Wm-1)<br>
  
  常見的 N-Gram 模型有 Unigram(1-gram)，Bigram(2-gram)，Trigram(3-gram)。<br>
  當N值愈大，對字詞的約束性愈大，具有愈高的辨識力，但同時複雜度也較高，需要更大的文本資料來訓練模型。<br>
  
  N-Gram 常用的應用場景像是 **”選字推薦”、”錯字勘正”、”分詞系統”** 等。**[作業-
以Bigram模型下判斷語句是否合理](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day8_%E5%9F%BA%E7%A4%8E%E8%AA%9E%E8%A8%80%E6%A8%A1%E5%9E%8BN_Gram%E4%BD%9C%E6%A5%AD.ipynb)**<br>
  

### <a name="C">③ NLP 詞性標註</a><br>
* #### 中文<br>
**1. CkipTagger：** from ckiptagger import POS -> pos = POS("./data") -> pos_sentence_list = pos(word_sentence_list) -> 取得中文的詞型標註序列(需先斷詞)。<br>
**2. jieba [作業-jieba詞性標註](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day11_jieba%E4%BD%9C%E6%A5%AD.ipynb)：** import jieba.posseg as pseg -> pseg.cut(sentence,) 取得中文的詞型標註。<br>
* #### 英文<br>
**1. NLTK [作業-詞幹詞條提取](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day13_%E8%A9%9E%E5%B9%B9%E8%A9%9E%E6%A2%9D%E6%8F%90%E5%8F%96%EF%BC%9AStemming%20and%20Lemmatization%E4%BD%9C%E6%A5%AD.ipynb)：** from nltk import pos_tag -> pos_tag(word_tokenize(sent)) -> get tuple (word, tag)

### <a name="D">④ 經典詞彙向量化(詞袋分析/ TF-IDF / SVD&共現矩陣)</a><br>
* #### Bag-of-words [作業-搭建一個bag of words模型](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day12_bag_of_words%E4%BD%9C%E6%A5%AD.ipynb)<br>
  文字向量長度 = 文本所有unique文字數。<br>
  先建立word和index的對應字典，計算文章向量時，抓出每個文章內的字，出現一次就在對應的index上+1。<br>
  文本Bag-of-words矩陣大小 = 文章數 X unique文字數。<br>
  
* #### TF-IDF [作業-搭建一個TFIDF 模型](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day15_tfidf%E4%BD%9C%E6%A5%AD.ipynb)<br>
  TF(詞頻)：一個單詞出現在一個文件的次數/該文件中所有單詞的數量<br>
  IDF： Log (所有文件的數目/包含這個單詞的文件數目)<br>
  TF-IDF的值代表文字對於文章的重要性（可藉此去除，或將一些常用詞權重降低）。<br>
  
* #### 共現矩陣→PPMI→SVD [作業-詞庫&共現矩陣的優缺點](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day16_%E8%A8%88%E6%95%B8%E6%96%B9%E6%B3%95%E8%A9%9E%E5%90%91%E9%87%8F%E4%BB%8B%E7%B4%B9_%E4%BD%9C%E6%A5%AD.ipynb)<br>
  分布假說：假設字詞本身沒有意義，字詞意義是根據該詞的”上下文（context）”形成的。<br>
   →可以透過計數周圍(window)的字詞來表達特定字詞的向量(共現矩陣)。<br>
   
  **共現矩陣實現 [作業-計數方法詞向量實作](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day17_%E8%A8%88%E6%95%B8%E6%96%B9%E6%B3%95%E8%A9%9E%E5%90%91%E9%87%8F%E5%AF%A6%E4%BD%9C_%E4%BD%9C%E6%A5%AD.ipynb)：**<br>
  設置window，若window=2，在製作共現矩陣時則會將中間字的**前後各兩個字** +1(或是依照和中間字的距離加上計算後的權重數字)<br>
  但共現矩陣有2個缺點：<br>
  ◎維度龐大<br>
  ◎對高頻詞(常用詞)效果差<br>
  改善方法：<br>
  ◎用 **SVD奇異值分解(scikit-learn TruncatedSVD)** 降維。<br>
  ◎計算 **PPMI(正向點間互資訊)** 將字詞出現頻率 P(x),P(y),P(x,y) 納入考慮，改善高頻詞效果差的問題<br>
  利用 **scikit-learn TruncatedSVD** 時可使用 **explained_variance_ratio_.sum()** 了解降維後的資料大約呈現多少比例的原始資料。<br>
  
* #### 詞幹/詞條提取：Stemming and Lemmatization(英文) [作業-詞幹詞條提取](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day13_%E8%A9%9E%E5%B9%B9%E8%A9%9E%E6%A2%9D%E6%8F%90%E5%8F%96%EF%BC%9AStemming%20and%20Lemmatization%E4%BD%9C%E6%A5%AD.ipynb)<br>
  詞幹/詞條提取是將單詞的不同型態歸一化(Ex: makes made making -> make)，藉此來降低文本的複雜度。<br>
  常見作法：<br>
**1. Stemming**<br>
  ★ 較為簡單的作法，制定規則來拆解單詞(rule-based)，像是看到 ing/ed 就去除。<br>
  ★ 問題 → Overstemming(造成多個不同單字變成相同單字，使單字失去原意。) / Understemming(去除不夠乾淨，使單字失去原意，甚至會製造出多的意義不明的字。)<br>
**2. Lemmatization**<br>
  ★ 需要有字典來尋找單詞的原型，也可以利用 **POS tagging** 的訊息來給出更正確的答案。<br>
   → Ex: 'lemmatization amusing : {}'.format(lemmatizer.lemmatize('amusing',pos = 'v'))  -->也可以不加pos<br>
  ★ 通常Lemmatization的效果較好，但也較花時間。<br>
  
  ※現在有單詞拆解技術例如BERT的Wordpiece等，所以Stemming和Lemmatization已較少使用。<br>
  
### <a name="E">⑤ NLP 經典機器學習模型(scikit-learn)</a><br>
* #### 機器學習模型</font><br>

**1. KNN [作業-KNN實作](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day20_21_KNN%E5%AF%A6%E4%BD%9C%E4%BD%9C%E6%A5%AD.ipynb)**<br>

  ★ 適合資料少、維度小的資料。<br>
  ★ k值過大易underfitting;k值過小易overfitting。<br>
  ★ k可以先設定成k\*\*0.5再進行調整，且應盡量設成奇數(防止有票數相同無法判別類別的問題)。<br>

**2. Naïve Bayes [作業-Naive_Bayes實作](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day23_Naive_Bayes%E5%AF%A6%E4%BD%9C%E4%BD%9C%E6%A5%AD.ipynb)**<br>

  ★ 為什麼naïve？   →    假設所有的輸入特徵都是**彼此獨立**、且**不管順序**。<br>
  ★ scikit-learn API比較↓
| **scikit-learn API** |**Note** |
| :-------------: | :-----:|
| Naïve Bayes Multinomial|特徵為離散型資料可使用 |
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

**4. Random Forest [作業-實作隨機森林](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day27_%E5%AF%A6%E4%BD%9C%E6%A8%B9%E5%9E%8B%E6%A8%A1%E5%9E%8B_%E4%BD%9C%E6%A5%AD.ipynb)**<br>

  ★ Decision Tree 的集成學習(Ensemble learning) Bagging。<br>
  ★ 假設每個分類氣兼具有差異 & 每個分類自單獨表現夠好(Accuracy>0.5)<br>
  ★ 隨機抽選樣本和特徵，相較Decision Tree不易overfitting。<br>
  ★ 抗noise力較好(採隨機抽樣)，泛化性較佳。<br>
  ★ 對資料量小 & 特徵少的資料效果不佳。<br>

**5. Adaboost [作業-用樹型模型進行文章分類](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day28_%E5%AF%A6%E4%BD%9CTreeBase%E6%A8%A1%E5%9E%8B_%E4%BD%9C%E6%A5%AD%20.ipynb)**<br>

  ★ Decision Tree 的集成學習(Ensemble learning) Boosting。<br>
  ★ 將模型以序列的方式串接，透過過加強學習前一步的錯誤，來增強這一步模型的表現。<br>
  ★ 對noise敏感。<br>
  ★ 訓練時間較長。<br>

**※Random Forest & Adaboost 比較**<br>
| **Random Forest** |**Adaboost** |
| :-----: | :-----:|
| 每個Decision Tree都是Full sized，不會事先決定最大分支數| 每個Decision Tree通常只有一個節點&2片葉(stump)|
| 會使用多個variables進行決策| 一次只用一個variables | 
| 最終決策時，採均勻多數決|最終決策時，每棵樹票不一樣大|
| 樹的順序無影響|樹間有關連(序列架構)，順序很重要|
| 泛化能力佳 抗noise能力較高| 對noise敏感 |
| 使用均勻採樣bootstrap(抽中放回)|上個模型分錯的樣本，下次被抽中的機率較高|

**※樹型(Tree base)模型衡量指標**<br>
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

**1. KFold [作業 實現K-fold分割資料](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day19_K_fold%E4%BD%9C%E6%A5%AD.ipynb)**<br>

  將**訓練集切成K份**,其中K-1份當作訓練資料，而1份則作為驗證資料。<br>
  在scikit-learn中可以用以下方式來實現：
  1. cross_val_score(clf, X, y, cv=5) ※clf=分類模型、cv=分成幾份(K)<br>
  2. KFold(n_splits=10, random_state=None, shuffle=False) shuffle=True → 按順序切分<br>
  
* #### Bias-Variance Tradeoff [作業-了解何謂Bias-Variance Tradeoff](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day25_Random_forest%E4%BD%9C%E6%A5%AD.ipynb)<br>

一般模型誤差可拆為三部分<br>
  →   **整體誤差(Total error)= 偏差(Bias) + 變異(variance) + 隨機誤差(random error => noise)**<br>

**1. Bias**： 訓練集的預測結果和實際答案的落差。<br>
**2. Variance**： 模型在測試資料集的表現 → 模型的泛化性指標。<br>

  ★ **Bias過大 → underfitting** ， 但 **Bias小 Variance過大 → overfitting**。<br>
  ★ Bias-Variance Tradeoff就是在Bias 和 Variance 中取得一個最佳的平衡，目的是使**整體誤差下降**。<br>
  
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
