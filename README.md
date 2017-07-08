# skip-thought vectors

## 跳躍思考ベクトル
 skip-thougt vectorはword2vecなどが、周辺の単語から自己の情報を規定するのに対して、自分の文章が他の文章から、どのように構築されているかを把握するものです  
 
 仕組みは簡単で、周辺文脈までSeq2SeqベースのAutoEncoderでモデルを構築することができます  
 
 連続する三つの文章があったとします。それぞれを順番順に、S1, S2, S3とします。Seq2SeqなどのEncoder-Decoderモデルで、S2から、S1,S3の文章を予想します。
 この時、S2からS1,S3に情報を受け渡すときに、ベクトル情報が渡されます。  
 
 <p align="center">
   <img width="700px" src="https://user-images.githubusercontent.com/4949982/27987989-75c479a2-6452-11e7-9f08-4c827f9909a3.png">
 </p>
 <div align="center"> 図1. 元論文の図解 </div>
 
 このベクトルが文章の前後を予想するベクトル、つまり、skip-thought vectorが出力されます。
 
 このベクトルは特徴的な特性があり、単語を組み合わせて意味を取ろうとしたら難しい意味をうまくとることができます。（下図参照）
  <p align="center">
   <img width="700px" src="https://user-images.githubusercontent.com/4949982/27988042-b40cdaf0-6453-11e7-82fd-90fd7096a70a.png">
 </p>
 <div align="center"> 図2. 少女がコスチュームを来た女性を見ていると、コスチュームを着た少女が女性を見ているがあまり似ていない</div>
 
 ## Kerasでのモデルの構築
  Kerasで元論文の意図を組んで実装しました。  
  
<p align="center">
  <img width="700px" src="https://user-images.githubusercontent.com/4949982/27987905-dbc9f3a0-6450-11e7-8c17-2866ff8ade9e.png">
</p>
<div align="center"> 図3. 今回の実装の概要図 </div>
  単語粒度での実装と、onehotベクトルでなくて、fasttextによる分散表現を用いています  
  
  分散表現で意味の類似度を効力した単語ベクトルになるので、skip-though vectorsの学習で用いなかった単語などにも対応が可能ということが論文で示唆されいます    
  
### コード
参考になるかどうかわかりませんが、お役に立てたら幸いです  
fasttextの分散表現は256次元で、エンコード字の活性化関数は、論文に従いtanhを用いました  

[github](https://github.com/GINK03/keras-skip-though-vector)
```python3
WIDTH       = 256
ACTIVATOR   = 'selu'
DO          = Dropout(0.1)
inputs      = Input( shape=(20, WIDTH) ) 
encoded     = Bi( GRU(256, kernel_initializer='lecun_uniform', activation=ACTIVATOR, return_sequences=True) )(inputs)
encoded     = TD( Dense(512, kernel_initializer='lecun_uniform', activation=ACTIVATOR) )( encoded )
encoded     = TD( Dense(512, kernel_initializer='lecun_uniform', activation=ACTIVATOR) )( encoded )
encoded     = Flatten()( encoded )
encoded     = Dense(1024, kernel_initializer='lecun_uniform', activation=ACTIVATOR)( encoded )
encoded     = DO( encoded )
encoded     = Dense(1024, kernel_initializer='lecun_uniform', activation=ACTIVATOR)( encoded )
encoded     = Dense(256, kernel_initializer='lecun_uniform', activation='tanh')( encoded )
encoder     = Model(inputs, encoded)

decoded_1   = Bi( GRU(256, kernel_initializer='lecun_uniform', activation=ACTIVATOR, return_sequences=True) )( RepeatVector(20)( encoded ) )
decoded_1   = TD( Dense(256) )( decoded_1 )
decoded_1   = TD( Dense(256) )( decoded_1 )

decoded_2   = Bi( GRU(256, kernel_initializer='lecun_uniform', activation=ACTIVATOR, return_sequences=True) )( RepeatVector(20)( encoded ) )
decoded_2   = TD( Dense(256) )( decoded_1 )
decoded_2   = TD( Dense(256) )( decoded_1 )

skipthought = Model( inputs, [decoded_1, decoded_2] )
skipthought.compile( optimizer=Adam(), loss='mean_squared_logarithmic_error' )
```
 

