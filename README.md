## Domain Proyek

Domain Proyek ini adalah Keuangan, dengan melakukan  Predictive Analytics harga emas yang ada di pasar komoditas

Orang yang membeli emas memiliki tujuan salah satunya adalah untuk berinvestasi karena harga emas cenderung selalu naik. cara pembeliannya melalui  cara tradisional dengan membeli perhiasan atau dengan cara modern seperti membeli koin/batangan emas atau dengan berinvestasi di Gold Dana yang diperdagangkan di bursa/forex.

Forex termasuk investasi beresiko tinggi karena transaksi yang kurang tepat dapat langsung mengambil modal dalam sebuah akun dengan cepat. Oleh karena itu, para trader harus mengetahui kapan harus beli/jual dan berapa lama menunggu. Salah satu cara yang dapat dilakukan adalah dengan menggunakan teknik *forecasting* atau peramalan dimasa akan datang dengan menggunakan data-data yang telah ada di masa lalu untuk memprediksi data yang akan datang.

## Business Understanding

### Problem Statement
Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut :
* Bagaimana menganalisa data harga *Emas*?

### Goals
Tujuan proyek ini dibuat antara lain sebagai berikut :
* Dapat memprediksi harga *Emas* dengan menggunakan model machine learning agar dapat Membantu *trader* dalam melakukan pembelian pada *Emas*.

### Solution Statement
Solusi yang dapat dilakukan agar goals terpenuhi adalah sebagai berikut :
* Melakukan analisa, eksplorasi, pemrosesan pada data sampai dengan  memvisualisasikan data agar mendapatkan gambaran datanya tersebut. 
* Menangani *missing value* pada data.
* Mencari korelasi pada data untuk mencari *dependant variable* dan *independent variable*.
* Menangani outlier pada data dengan menggunakan Metode IQR.
* Melakukan normalisasi pada data terutama pada fitur numerik.
* Membuat model regresi untuk memprediksi bilangan kontinu untuk memprediksi harga yang akan datang.     
* Algoritma yang digunakan pada proyek ini :
    * Support Vector Machine (Support Vector Regression)
    * K-Nearest Neighbors
    * Boosting Algorithm (Gradient Boosting Regression)    
* Melakukan hyperparameter tuning agar model dapat berjalan pada performa terbaik dengan menggunakan teknik Grid Search

## Data Understanding

Dataset yang digunakan pada proyek ini adalah dataset dari Kaggle [https://www.kaggle.com/datasets/mattiuzc/commodity-futures-price-history]

Dataset yang digunakan memiliki format *.csv* yang mempunyai total 5291 record dengan 7 kolom (*Date, Open, High, Low, Close, Adj Close, Volume*)  Memiliki total 112 record  *missing value* pada masing-masing kolom *Open, High, Low, Close, Adj Close, Volume* dengan informasi sebagai berikut :
  * Date : Tanggal pencatatan Data
  * Open : Harga buka dihitung perhari
  * High : Harga tertinggi perhari
  * Low : Harga terendah perhari
  * Close : Harga tutup dihitung perhari
  * Adj Close : Harga penutupan pada hari tersebut setelah disesuaikan dengan aksi korporasi seperti right issue, stock split atau stock reverse.
  * Volume : Volume transaksi

### Exploratory Data Analysis
Sebelum melakukan pemrosesan data, kita harus mengetahui keadaan data. dengan mencari korelasi antar fitur, mencari outlier dan melakukan analisis *univariate* dan *multivariate*.

- Menangani outlier
<br>
<image src='https://github.com/wahyoe99/prediksihargaemas/blob/0ecae980a236cd1f2aa31026f81a43329bd99dbc/gambar/outlier.png' width= 500/>
<br>
Jika data numerik divisualisasikan, hanya fitur *Volume* saja yang memiliki outlier. Untuk menangani outlier kita akan menggunakan IQR Method yaitu dengan menghapus data yang berada diluar IQR yaitu antara 25% dan 75%. setelah melakukan kegiatan mengatasi outlier, didapatkan sampel 4550 record dan 7 Kolom.

- Univariate Analysis
<br>
<image src='https://github.com/wahyoe99/prediksihargaemas/blob/fe24d908a8d8254040603d167594f2e4b01c42be/gambar/Univariate.png' width= 500/>
<br>
Pada kasus ini kita hanya akan berfokus dalam memprediksi pada kolom *Adj Close*.

- Multivariate Analysis
<br>
<image src='https://github.com/wahyoe99/prediksihargaemas/blob/bd63205de20704301bd407fc2d1c62e9430d4f5d/gambar/Multivariate.png' width= 500/>
<br>
Selanjutnya kita akan menganalisis korelasi fitur *Adj Close* terhadap fitur lain seperti *Open, High, Low, Close dan Volume*. Dapat disimpulkan bahwa *Adj Close* memiliki korelasi positif yang kuat terhadap *Open, High, Low dan Close*, sedangkan untuk fitur *Volume* memiliki korelasi sedang terhadap fitur *Adj Close*.

- Korelasi
<br>
<image src='https://github.com/wahyoe99/prediksihargaemas/blob/bd63205de20704301bd407fc2d1c62e9430d4f5d/gambar/korelasi.png' width= 500/>
<br>
Untuk memperjelas korelasi kita akan memvisualisasikannya menggunakan heatmap dari library Seaborn. Dapat kita lihat bahwa *Adj Close* memiliki korelasi positif tinggi pada setiap fitur, kecuali fitur *Volume* sehingga kita dapat menggunakan semua fitur sebagai *dependant variable*.

    
## Data Preparation

Berikut merupakan tahapan-tahapan dalam melakukan pra-pemrosesan data:

### Menampilkan semua data dan kolom
Menampilkan semua data dan kolom dari dataset yang sudah di load

### Mengubah kolom date dari tipe object menjadi datatime
Melakukan perubahan tipe data object menjadi datetime agar datanya dapat diolah lebih mudah jika memang datanya berisi tanggal. seperti melakukan perhitungan selisih hari, bulan dan tahun.

### Melakukan Penanganan Missing Value
Dalam menangani Missing Value menggunakan library SimpleImputer, yang dimana library ini bertugas untuk mengisi kolom yang memiliki missing value dengan data mean
   
### Menghapus fitur yang tidak diperlukan
Karena kita tidak memerlukan kolom *Date* dan *Volume* kita akan menghapus fitur *Date* dan *Volume*. Juga kita tidak memerlukan fitur *Close* karena *Adj Close* lebih akurat dari pada *Close*

### Melakukan pembagian dataset
Dataset akan dibagi menjadi 2 yaitu sebagai train data dan test data. Train data digunakan sebagai training model dan test data digunakan sebagai validasi apakah model sudah akurat atau belum. Ratio yang umum dalam splitting dataset adalah 80:20, 80% sebagai train data dan 20% sebagai test data. Pembagian dataset dilakukan dengan modul train_test_split dari scikit-learn. Setelah melakukan pembagian dataset, didapatkan jumlah sample pada data latih sebanyak 3640 sampel dan jumlah sample pada data test yaitu 910 sampel dari total jumlah sample pada dataset yaitu 4550 sampel.

### Data Normalization
Normalisasi data digunakan agar model dapat bekerja lebih optimal karena model tidak perlu mengolah data dengan angka besar. Normalisasi biasanya mentransformasi data dalam skala tertentu. Untuk proyek ini kita akan normalisasi data 0 hingga 1 menggunakan MinMaxScaler.

## Modeling

Model yang akan digunakan proyek kali ini yaitu *Support Vector Regression, Gradient Boosting,* dan *K-Nearest Neighbors*.

### Support Vector Regression
*Support Vector Regression* memiliki prinsip yang sama dengan SVM, namun SVM biasa digunakan dalam klasifikasi. Pada SVM, algoritma tersebut berusaha mencari jalan terbesar yang bisa memisahkan sampel dari kelas berbeda, sedangkan SVR mencari jalan yang dapat menampung sebanyak mungkin sampel di jalan. Untuk hyper parameter yang digunakan pada model ini adalah sebagai berikut :
* *kernel* : Hyperparameter ini digunakan untuk menghitung kernel matriks sebelumnya.
* *C* : Hyperparameter ini adalah parameter regularisasi digunakan untuk menukar klasifikasi yang benar dari contoh *training* terhadap maksimalisasi margin fungsi keputusan.
* *gamma* : Hyperparameter ini digunakan untk menetukan seberapa jauh pengaruh satu contoh pelatihan mencapai, dengan nilai rendah berarti jauh dan nilai tinggi berarti dekat.

##### Kelebihan
* Lebih efektif pada data dimensi tinggi (data dengan jumlah fitur yang banyak)
* Memori lebih efisien karena menggunakan subset poin pelatihan

##### Kekurangan
* Sulit dipakai pada data skala besar

### K-Nearest Neighbors
*K-Nearest Neighbors* merupakan algoritma machine learning yang bekerja dengan mengklasifikasikan data baru menggunakan kemiripan antara data baru dengan sejumlah data (k) pada data yang telah ada. Algoritma ini dapat digunakan untuk klasifikasi dan regresi. Untuk hyperparameter yang digunakan pada model ini hanya 1 yaitu :
* *n_neighbors* : Jumlah tetangga untuk yang diperlukan untuk menentukan letak data baru

##### Kelebihan
* Dapat menerima data yang masih *noisy*
* Sangat efektif apabila jumlah datanya banyak
* Mudah diimplementasikan

##### Kekurangan
* Sensitif pada outlier
* Rentan pada fitur yang kurang informatif

### Gradient Boosting
Gradient Boosting adalah algoritma machine learning yang menggunakan teknik *ensembel learning* dari *decision tree* untuk memprediksi nilai. Gradient Boosting sangat mampu menangani pattern yang kompleks dan data ketika linear model tidak dapat menangani. Untuk hyperparameter yang digunakan pada model ini ada 3 yaitu :
* *learning_rate* : Hyperparameter training yang digunakan untuk menghitung nilai koreksi bobot padad waktu proses training. Umumnya nilai learning rate berkisar antara 0 hingga 1
* *n_estimators* : Jumlah tahapan boosting yang akan dilakukan.
* *criterion* : Hyperparameter yang digunakan untuk menemukan fitur dan ambang batas optimal dalam membagi data

##### Kelebihan
* Hasil pemodelan yang lebih akurat
* Model yang stabil dan lebih kuat (robust)
* Dapat digunakan untuk menangkap hubungan linear maupun non linear pada data

##### Kekurangan
* Pengurangan kemampuan interpretasi model
* Waktu komputasi dan desain tinggi
* Tingkat kesulitan yang tinggi dalam pemilihan model

Untuk proyek kali ini kita akan menggunakan model *K-Nearest Neighbors* karena memiliki error (*0.00001*) yang paling sedikit daripada model yang lain. Namun tidak bisa dipungkiri model dari Gradient Boosting juga memiliki error (*0.000011*) yang hampir seperti *KNN*.




## Evaluation

Untuk evaluasi pada machine learning model ini, metrik yang digunakan adalah *mean squared error (mse)* dan R2 _coefficient of determination_ . Dimana metrik ini mengukur seberapa dekat garis pas dengan titik data.
<br>
<image src='https://github.com/wahyoe99/prediksihargaemas/blob/bd63205de20704301bd407fc2d1c62e9430d4f5d/gambar/mse.jpg' width= 500/>
<br>

dimana :
n = jumlah titik data
Yi = nilai sesungguhnya
Yi_hat = nilai prediksi

![](https://cdn1.byjus.com/wp-content/uploads/2021/04/Correlation-formula.png)

Menampilkan hasil akurasi dari beberapa model yang dipakai :
<br>
<image src='https://github.com/wahyoe99/prediksihargaemas/blob/415c4d3c6e4f7ee45e88b5fb7d37d7a9c17291c0/gambar/modeling.png' width= 400/>
<br>
<image src='https://github.com/wahyoe99/prediksihargaemas/blob/415c4d3c6e4f7ee45e88b5fb7d37d7a9c17291c0/gambar/hasilmse.png' width= 400/>
<image src='https://github.com/wahyoe99/prediksihargaemas/blob/415c4d3c6e4f7ee45e88b5fb7d37d7a9c17291c0/gambar/hasilr2.png' width= 400/>
<br>
Untuk proyek kali ini terdapat 2 model yang dapat berjalan dengan performa optimal yaitu, *Gradient Boosting* model dan *K-Nearest Neighbors*. Terdapat selisih nilai yang sangat kecil. Tetapi pada perhitungan akurasi model terdapat model yang menggunakan *K-Nearest Neighbors* memiliki nilai lebih tinggi.

## Kesimpulan

Dari hasil evaluasi di atas dengan menggunakan MSE dan R2 atau coefficient of determination dapat memberikan informasi bahwa ketiga model yang dibangun memiliki performa di atas 99% mendekati 100%. Dimana dapat dilihat juga bahwa model dengan algoritma KNN memiliki performa yang diukur dengan nilai akurasi yang lebih baik dari dua model lainnya yaitu model dengan algoritma SVR dan Gradient Boost. Sehingga untuk dapat memprediksi harga emas selama 30 hari kedepan bisa menggunakan algoritma KNN karena secara akurasinya hampir mendekati sempurna 100%

Dari hasil perbandingan dengan menggunakan 3 model diatas didapatkan hasil yang terbaik untuk bisa melakukan analisa prediksi harga emas dengan menggunakan KNN karena dari evaluasinya akurasinya paling tinggi hampir mendekati angka 100%, dan bisa menjawab beberapa pertanyaan di Business Understanding
* Dapat menganalisa prediksi data harga *Emas* dengan menggunakan model KNN
* Tujuannya mencari model terbaik yang bisa digunakan untuk melakukan prediksi harga *Emas*
* Dan juga dapat membantu *trader* untuk dapat mengambil keputusan dalam melakukan pembelian pada *Emas*
* Solusi yang sudah dilakukan dari mulai prepare data sampai normalisasi dan membuat model regresi sampai membandingkan 3 algoritma yang digunakan untuk bisa mendapatkan akurasi yang terbaik
