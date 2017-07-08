# skip-thought vectors

## 跳躍思考ベクトル
 skip-thougt vectorはword2vecなどが、周辺の単語から自己の情報を規定するのに対して、自分の文章が他の文章から、どのように構築されているかを把握するものです  
 仕組みは簡単で、周辺文脈までSeq2SeqベースのAutoEncoderでモデルを構築することができます  
 
 連続する三つの文章があったとします。それぞれを順番順に、S1, S2, S3とします。Seq2SeqなどのEncoder-Decoderモデルで、S2から、S1,S3の文章を予想します。
 この時、S2からS1,S3に情報を受け渡すときに、ベクトル情報が渡されます。  
 
 <p align="center">
   <img width="700px" src="https://user-images.githubusercontent.com/4949982/27987657-0497d5cc-644c-11e7-9f90-c8923b9602e0.png">
 </p>
 <div align="center"> 図1. 元論文の図解 </div>
 
 このベクトルが文章の前後を予想するベクトル、つまり、skip-thought vectorが出力されます。
