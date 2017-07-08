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
 
 ## Kerasでのモデルの構築
  Kerasで元論文の意図を組んで実装しました。  
  
<p align="center">
  <img width="700px" src="https://user-images.githubusercontent.com/4949982/27987905-dbc9f3a0-6450-11e7-8c17-2866ff8ade9e.png">
</p>
<div align="center"> 図2. 今回の実装の概要図 </div>
  単語粒度での実装と、onehotベクトルでなくて、fasttextによる分散表現を用いています  
  
  分散表現で意味の類似度を効力した単語ベクトルになるので、skip-though vectorsの学習で用いなかった単語などにも対応が可能ということが論文で示唆されいます    
  

 

