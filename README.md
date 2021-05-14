# NLP_ML_Marathon
## NLP經典機器學習馬拉松 6大學習里程碑◢

**[① Python NLP 程式基礎](#A)**<br>

**[② 詞彙與分詞技術](#B)**<br>

**[③ NLP 詞性標註](#C)**<br>

**[④ 經典詞彙向量化(詞袋分析/ TF-IDF / SVD&共現矩陣)](#D)**<br>

**[⑤ NLP 經典機器學習模型](#E)**<br>

**[⑥ 期末實務專題](#F)**<br>


## 學習重點整理

### <a name="A">① Python NLP 程式基礎</a><br>
* #### 文字處理函數<br>
**[作業 String operation1](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day1-%20String%20operation%E4%BD%9C%E6%A5%AD.ipynb)**<br>
**[作業 String operation2](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day2-%20String%20operation%E4%BD%9C%E6%A5%AD.ipynb)**<br>

* #### 正規表達式<br>
★測試pattern可至 **[Regex101](https://regex101.com/)**<br>

**[作業 利用正規表達式達到預期配對](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day3_Regex_%E4%BD%9C%E6%A5%AD%20.ipynb)**<br>
**[作業 使用python正規表達式對資料進行清洗處理](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day4_Python_regular_expression_%E4%BD%9C%E6%A5%AD%20.ipynb)**<br>

### <a name="B">② 詞彙與分詞技術</a><br>
* #### 斷詞的方法--經典中文斷詞法簡介(jieba)<br>
**1. 針對存在於字典的字詞**：<br>
     **Trie樹  -> 給定輸入句的 DAG -> 動態規劃(Dynamic Programming) 來找出最大機率路徑**<br>

  ★ **Trie樹：** 以Root(不包含字符)為起始往下延伸節點，每個節點代表一個字，連接的下個節點代表字詞中的下一個字。<br>
   → Ex: tea & ted 的 Trie樹 可以表示成 Root -> t -> e -> a、d (a 和 d 各自成 1 個節點，前面的 Root、t、e 共用) <br>
   
  ★ **DAG (有向無環圖)：** 收到待分詞句子後**查詢Trie樹，列出所有可能的切分結果**。<br>
  在jieba中以字典形態呈現，詞的開始位置做為Key，並以所有的可能結尾位置(list)做為value。<br>
  
  ★ **動態規劃(Dynamic Programming)：** 根據jieba字典中的詞頻的**最大機率路徑**得到最後斷詞結果。
    
**2. 針對不存在於字典的字詞：**<br>
     **隱馬可夫模型(HMM) & 維特比演算法(Viterbi) [作業-Viterbi實作](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day5_%E6%96%B7%E8%A9%9E%E4%BD%9C%E6%A5%AD%20.ipynb)**<br>

   ★ **<a name="M">馬可夫模型： 從目前狀態轉移 s 到下一個狀態 s' 的機率由 P(s'|s) 來決定 (在 s 的前提下 s’ 發生的機率)</a>**<br>
   
   **◎一階馬可夫模型：**當前狀態**只與前一個狀態**有關。 P(Xi|Xi-1)<br>
   **◎m階馬可夫模型：**當前狀態可能受**前 m 個狀態所影響**。P(Xi|Xi-1, Xi-2, ...., Xi-m)<br>
    
   >計算時的向量矩陣<br>
   **初始機率向量(PI 向量)：** 每個狀態的初始機率所構成的向量(總和為1)。<br>
   **狀態轉移矩陣：** 從1個狀態轉移到下一個狀態時的機率。<br>
   → 1個 m階馬可夫模型的轉移矩陣大小會是 **m\*m** (每個狀態都有可能是其他狀態的下一個狀態)。<br>
   
   ★ **隱馬可夫模型(HMM)： 當無法直接觀察到的隱藏狀態時，利用可觀測到的觀察狀態，來推測隱藏狀態的機率。**<br>
   
   >計算時的向量矩陣<br>
   **隱藏狀態：** 系統的真實狀態(想預測的狀態)。<br>
   **觀察狀態：** ”可觀測"的狀態。<br>
   **初始機率向量(PI 向量)：** 隱藏狀態的初始機率所構成的向量(總和為1)。<br>
   **狀態轉移矩陣：** 從1個隱藏狀態轉移到下一個隱藏狀態的機率。<br>
   **發射矩陣：** 隱藏狀態觀察到某一個觀察狀態的機率。<br>
   
   ★ **維特比(Viterbi)： 使用動態規劃求解隱馬可夫模型預測(求解最大機率路徑)。**<br>
   
   ※jieba未知詞斷詞是使用**HMM**來判斷斷詞結果。<br>
   >**隱藏狀態：** {B:begin, M:middle, E:end, S:single}。<br>
   **觀察狀態：** 所有的詞(包含標點符號)。<br>
   **初始機率向量(PI 向量)：**  內建 “prob_start.py”。<br>
   **狀態轉移矩陣：** 內建 “prob_trans.py”，維度大小(BEMS x BEMS)。<br>
   **發射矩陣：** 內建 “prob_emit.py”。<br>
   
  
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
**使用[馬可夫假設](#M)簡化**<br>
  
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
  
* #### 延伸：其他Ensemble learning [作業-了解Ensemble中的Blending與Stacking](https://github.com/hyh009/NLP_ML_Marathon/blob/master/Day26_Adaboost%E4%BD%9C%E6%A5%AD.ipynb)<br>
**1. Stacking<br>**
**2. Blending<br>**

### <a name="F">⑥ 期末實務專題</a><br>

* #### [自製中文選字系統](https://github.com/hyh009/NLP_ML_Marathon/tree/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(1)%20%E8%87%AA%E8%A3%BD%E4%B8%AD%E6%96%87%E9%81%B8%E5%AD%97%E7%B3%BB%E7%B5%B1)<br>
1. [基礎篇](https://github.com/hyh009/NLP_ML_Marathon/blob/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(1)%20%E8%87%AA%E8%A3%BD%E4%B8%AD%E6%96%87%E9%81%B8%E5%AD%97%E7%B3%BB%E7%B5%B1/%E5%9F%BA%E7%A4%8E%E7%AF%87_hw%20.ipynb)<br>

**使用N-Gram製作中文選字系統**。<br>

實作重點整理：<br>
★ 計算出文本的 n (分子) 和 n-1(分母)的所有字詞次數，並建立字典(利用Counter())。<br>
★ prefix取n-1個字->當作輸入值用來預測第n個字。<br>
★ 將 n個字詞的字典中 prefix 開頭的字詞出現的次數 / n-1個字詞字典中 prefix 出現的次數(獲得機率)，並將結果存進list中，按機率大小順序排好。<br>
★ return 機率最大的 k 個字。<br>

2. [進階篇](https://github.com/hyh009/NLP_ML_Marathon/blob/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(1)%20%E8%87%AA%E8%A3%BD%E4%B8%AD%E6%96%87%E9%81%B8%E5%AD%97%E7%B3%BB%E7%B5%B1/%E9%80%B2%E9%9A%8E%E7%AF%87_hw.ipynb)<br>

基礎篇延伸 + **Smoothing of Language Models**<br>
**★ Back-off Smoothing ：**<br>
當前 N-Gram (Ex:Trigram) 預測字的機率為0時，使用 (N-1)-Gram (Ex:bigram) 的機率乘上 alpha (alpha < 1)，以此類推 (alpha值取0.4為大宗)。<br>
**★ Interpolation Smoothing**<br>
預測使用到 **unigram ~ 指定的N-Gram 的機率**，但 最大的N-Gram (Ex:Trigram) 機率會乘上lambda，再加上 (N-1)-Gram (Ex:bigram)機率則乘上(1-lambda)，以此類推。<br>

實作重點整理：<br>
★ 定義backoff 和 interpolation的函式，藉由呼叫函式達到遞迴效果。<br>
★ 先取出unigram的所有Key值(=所有可能的字)，再利用迴圈計算每個字的機率。<br>

* #### [建置新聞分類器](https://github.com/hyh009/NLP_ML_Marathon/tree/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(2)%20%E5%BB%BA%E8%A3%BD%E6%96%B0%E8%81%9E%E5%88%86%E9%A1%9E%E5%99%A8)<br>

1. [基礎篇](https://github.com/hyh009/NLP_ML_Marathon/blob/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(2)%20%E5%BB%BA%E8%A3%BD%E6%96%B0%E8%81%9E%E5%88%86%E9%A1%9E%E5%99%A8/%E5%88%86%E9%A1%9E%E5%99%A8_%E5%9F%BA%E7%A4%8E%E7%AF%87_hw%20.ipynb)<br>

**使用bag of words & cosine similarity 建置新聞分類器**。<br>

實作重點整理：<br>
★ 利用詞性標註排除較無意義的詞性。<br>
★ 計算Group mean vector做為比較的基準。<br>
★ 利用cosine similarity計算每篇文章和每個Group mean vector的距離，並找出距離最近的類別。<br>

2. [進階篇](https://github.com/hyh009/NLP_ML_Marathon/blob/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(2)%20%E5%BB%BA%E8%A3%BD%E6%96%B0%E8%81%9E%E5%88%86%E9%A1%9E%E5%99%A8/%E5%88%86%E9%A1%9E%E5%99%A8_%E9%80%B2%E9%9A%8E%E7%AF%87_hw.ipynb)<br>

**使用TFIDF & cosine similarity 建置新聞分類器 + PCA視覺化**。<br>

實作重點整理：<br>
★ 使用PCA降維預備視覺化時，為了看清楚分類結果的分布，選擇用Group mean vector來fit模型。<br>

3. [PPMI+SVD](https://github.com/hyh009/NLP_ML_Marathon/blob/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(2)%20%E5%BB%BA%E8%A3%BD%E6%96%B0%E8%81%9E%E5%88%86%E9%A1%9E%E5%99%A8/%E5%88%86%E9%A1%9E%E5%99%A8%EF%BC%9APPMI%EF%BC%8BSVD_hw.ipynb)<br>

**使用共現矩陣+PPMI+SVD & cosine similarity 建置新聞分類器 + PCA視覺化**。<br>

* #### [垃圾郵件偵測器](https://github.com/hyh009/NLP_ML_Marathon/tree/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(3)%20%E6%96%87%E4%BB%B6%E5%88%86%E9%A1%9E%EF%BC%9A%E5%9E%83%E5%9C%BE%E9%83%B5%E4%BB%B6%E5%81%B5%E6%B8%AC%E5%99%A8%20(Spam%20Detector))<br>

1. [基礎篇](https://github.com/hyh009/NLP_ML_Marathon/blob/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(3)%20%E6%96%87%E4%BB%B6%E5%88%86%E9%A1%9E%EF%BC%9A%E5%9E%83%E5%9C%BE%E9%83%B5%E4%BB%B6%E5%81%B5%E6%B8%AC%E5%99%A8%20(Spam%20Detector)/%E5%9F%BA%E7%A4%8E%E7%AF%87%20spam_nb_%E5%9E%83%E5%9C%BE%E9%83%B5%E4%BB%B6%E5%81%B5%E6%B8%AC%E5%99%A8(%E4%BD%9C%E6%A5%AD).ipynb)

**使用ML(scikit-learn)方式分類辨別是否為垃圾郵件**。<br>

2. [進階篇](https://github.com/hyh009/NLP_ML_Marathon/blob/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(3)%20%E6%96%87%E4%BB%B6%E5%88%86%E9%A1%9E%EF%BC%9A%E5%9E%83%E5%9C%BE%E9%83%B5%E4%BB%B6%E5%81%B5%E6%B8%AC%E5%99%A8%20(Spam%20Detector)/%E9%80%B2%E9%9A%8E%E7%AF%87%20spam_sms_%E5%9E%83%E5%9C%BE%E7%B0%A1%E8%A8%8A%E5%81%B5%E6%B8%AC%E5%99%A8(%E4%BD%9C%E6%A5%AD).ipynb)

**進行文字預處理並利用ML(scikit-learn)分類辨別是否為垃圾郵件**。<br>

* #### [情緒分析](https://github.com/hyh009/NLP_ML_Marathon/tree/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(4)%20%E6%96%87%E4%BB%B6%E5%88%86%E9%A1%9E%EF%BC%9A%E7%94%A2%E5%93%81%E8%A9%95%E5%88%86%E6%83%85%E7%B7%92%E5%88%86%E6%9E%90(Sentiment%20Analysis))<br>

**以ML方式分辨電商產品評分文件為正向或負向**<br>

實作重點整理：<br>
★ 使用LogisticRegression時可以利用 coef_ function 查看每個字的正負權重。<br>

* #### [潛在語意分析](https://github.com/hyh009/NLP_ML_Marathon/tree/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(5)%20%E6%BD%9B%E5%9C%A8%E8%AA%9E%E6%84%8F%E5%88%86%E6%9E%90(Latent%20Semantics%20Analysis))<br>

**用SVD從文章題目視覺化文字的潛在語意**<br>

實作重點整理：<br>
★ 因為要視覺化的是文字，在製作vector時是製作文字的vector，維度會是文章的數量。<br>
★ 將代表文字的vector降至2維或3維即可視覺化。<br>

* #### [自動文件修改器(Trigram運用)](https://github.com/hyh009/NLP_ML_Marathon/tree/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(6)%20%E4%B8%89%E9%80%A3%E8%A9%9E(Trigram)%E4%B9%8B%E6%87%89%E7%94%A8)<br>

**利用 Trigram 建立自動文件修改器**<br>

實作重點整理：<br>
★ 取得文本的 Trigram 並建立字典，以第1和第3個字設為Key，第2個字設為value(可能有多個)。<br>
★ 利用迴圈計算每個Key值的各個value出現機率，並建立一個新字典 Key = (1,3)--個字，value = dict\[middle_word] = word_proba (此字典也可能有多個Key)。<br>
★ 設一個機率，修改原文的字(隨機挑選)。<br>

* #### [聊天機器人(單輪對話)](https://github.com/hyh009/NLP_ML_Marathon/tree/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(7)%20Rule-based%20chatbot%20(%E5%96%AE%E8%BC%AA%E5%B0%8D%E8%A9%B1))<br>

**建立一個單輪聊天機器人可以閒聊or回答問題**<br>

實作重點整理：<br>
★ 閒聊方式1：自定義Q&A(完全比對、部分比對、模糊比對)。<br>
★ 閒聊方式2：對青雲客 API 進行requests。<br>
★ 閒聊方式3：以Dcard做為語料，Key為文章標題、value為留言，並利用Fuzzy Chinese套件進行模糊比對query。<br>
★ 知識問答：Google搜尋並取得百科處的文字。<br>
★聊天機器人answer selection：使用斷詞&詞性標註技術查看query有幾個名詞、動詞、形容詞等，來判斷是閒聊還是知識問答。<br>
  →作業實作順序：先查看是否為自定義Q&A→是否有相似的Dcard標題→判斷閒聊 or 知識問答。<br>

* #### [Line Bot 聊天機器人(多輪對話)](https://github.com/hyh009/NLP_ML_Marathon/tree/master/%E6%9C%9F%E6%9C%AB%E5%AF%A6%E5%8B%99%E5%B0%88%E9%A1%8C(8)%20Line%E8%81%8A%E5%A4%A9%E6%A9%9F%E5%99%A8%E4%BA%BA%20(%E5%A4%9A%E8%BC%AA%E6%83%85%E5%A2%83))<br>

**在可查看高鐵班次的Line Bot多輪聊天機器人程式中新增查看股價功能**<br>

實作重點整理：<br>
★ 利用證券交易所爬蟲獲取台灣上市櫃公司名稱，再利用yahoo的yfinance套件獲取股價資訊。<br>
★ 利用 if elif判斷式建立聊天機器人程式。<br>
