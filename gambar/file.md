- Menangani outlier
<br>
<image src='https://raw.githubusercontent.com/Hanifanta/Gold_Predictive_Analytics/main/images/before_iqr.png' width= 500/>
<br> Jika data numerik divisualisasikan, hanya fitur *Volume* saja yang memiliki outlier. Untuk menangani outlier kita akan menggunakan IQR Method yaitu dengan menghapus data yang berada diluar IQR yaitu antara 25% dan 75%. setelah melakukan kegiatan mengatasi outlier, didapatkan sampel 4550 Data dan 7 Kolom.

- Univariate Analysis
<br>
<image src='https://raw.githubusercontent.com/Hanifanta/Gold_Predictive_Analytics/main/images/univariate.png' width= 500/>
<br> Pada kasus ini kita hanya akan berfokus dalam memprediksi *Adj Close*.

- Multivariate Analysis
<br> Selanjutnya kita akan menganalisis korelasi fitur *Adj Close* terhadap fitur lain seperti *Open, High, Low, Close dan Volume*. Dapat disimpulkan bahwa *Adj Close* memiliki korelasi positif yang kuat terhadap *Open, High, Low dan Close*, sedangkan untuk fitur *Volume* memiliki korelasi sedang terhadap fitur *Adj Close*.
<br>
<image src='https://raw.githubusercontent.com/Hanifanta/Gold_Predictive_Analytics/main/images/multivariate.png' width= 500/>


<br> Untuk memperjelas korelasi kita akan memvisualisasikannya menggunakan heatmap dari library Seaborn. Dapat kita lihat bahwa *Adj Close* memiliki korelasi positif tinggi pada setiap fitur, kecuali fitur *Volume* sehingga kita dapat menggunakan semua fitur sebagai *dependant variable*.
<br>
<image src='https://raw.githubusercontent.com/Hanifanta/Gold_Predictive_Analytics/main/images/heatmap.png' width= 500/>