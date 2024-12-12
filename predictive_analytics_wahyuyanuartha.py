# -*- coding: utf-8 -*-
"""predictive_analytics_wahyuyanuartha.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1J9gmztGhWEaC3grXZXx1hVmCVwmtqX8b

## **Domain Proyek**
---

Domain Proyek ini adalah Keuangan, dengan melakukan  Predictive Analytics harga emas yang ada di pasar komoditas

Orang yang membeli emas memiliki tujuan salah satunya adalah untuk berinvestasi karena harga emas cenderung selalu naik. cara pembeliannya melalui  cara tradisional dengan membeli perhiasan atau dengan cara modern seperti membeli koin/batangan emas atau dengan berinvestasi di Gold Dana yang diperdagangkan di bursa/forex.

Forex termasuk investasi beresiko tinggi karena transaksi yang kurang tepat dapat langsung mengambil modal dalam sebuah akun dengan cepat. Oleh karena itu, para trader harus mengetahui kapan harus beli/jual dan berapa lama menunggu. Salah satu cara yang dapat dilakukan adalah dengan menggunakan teknik *forecasting* atau peramalan dimasa akan datang dengan menggunakan data-data yang telah ada di masa lalu untuk memprediksi data yang akan datang.

## Business Understanding

### Problem Statement
Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut :
* Bagaimana menganalisa data harga *Emas*?


### Goals
Tujuan proyek ini dibuat antara lain sebagai berikut :
* Dapat memprediksi harga *Emas* dengan menggunakan model machine learning.
* Melakukan analisa dan mengolah data yang optimal agar model dapat Membantu para *trader* dalam melakukan pembelian pada *Emas*.

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

## **Data Understanding**
---

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
Jika data numerik divisualisasikan, hanya fitur *Volume* saja yang memiliki outlier. Untuk menangani outlier kita akan menggunakan IQR Method yaitu dengan menghapus data yang berada diluar IQR yaitu antara 25% dan 75%. setelah melakukan kegiatan mengatasi outlier, didapatkan sampel 4550 record dan 7 Kolom.

- Univariate Analysis
Pada kasus ini kita hanya akan berfokus dalam memprediksi pada kolom *Adj Close*.

- Multivariate Analysis
Selanjutnya kita akan menganalisis korelasi fitur *Adj Close* terhadap fitur lain seperti *Open, High, Low, Close dan Volume*. Dapat disimpulkan bahwa *Adj Close* memiliki korelasi positif yang kuat terhadap *Open, High, Low dan Close*, sedangkan untuk fitur *Volume* memiliki korelasi sedang terhadap fitur *Adj Close*.


Untuk memperjelas korelasi kita akan memvisualisasikannya menggunakan heatmap dari library Seaborn. Dapat kita lihat bahwa *Adj Close* memiliki korelasi positif tinggi pada setiap fitur, kecuali fitur *Volume* sehingga kita dapat menggunakan semua fitur sebagai *dependant variable*.
"""

#copy dataset dari kaggle
!kaggle datasets download -d mattiuzc/commodity-futures-price-history

# Commented out IPython magic to ensure Python compatibility.
#load library yang dibutuhkan
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
# %matplotlib inline

import zipfile

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#extract dataset yang sudah di copy diatas ke dalam googlecolab dan folder content
local_zip = '/content/commodity-futures-price-history.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

#tampilakan semua datanya
df = pd.read_csv('/content/Commodity Data/Gold.csv')
df

print(f'Jumlah data ada {df.shape[0]} records and {df.shape[1]} columns.')

df.info()

"""Deskripsi Variabel Numerik"""

df.describe()

"""# **Data Preparation**
---

Berikut merupakan tahapan-tahapan dalam melakukan pra-pemrosesan data:
### Melakukan Penanganan Missing Value
Dalam menangani Missing Value menggunakan library SimpleImputer, yang dimana library ini bertugas untuk mengisi kolom yang memiliki missing value dengan data mean

### Menghapus fitur yang tidak diperlukan
Karena kita tidak memerlukan kolom *Date* dan *Volume* kita akan menghapus fitur *Date* dan *Volume*. Juga kita tidak memerlukan fitur *Close* karena *Adj Close* lebih akurat dari pada *Close*

### Melakukan pembagian dataset
Dataset akan dibagi menjadi 2 yaitu sebagai train data dan test data. Train data digunakan sebagai training model dan test data digunakan sebagai validasi apakah model sudah akurat atau belum. Ratio yang umum dalam splitting dataset adalah 80:20, 80% sebagai train data dan 20% sebagai test data. Pembagian dataset dilakukan dengan modul train_test_split dari scikit-learn. Setelah melakukan pembagian dataset, didapatkan jumlah sample pada data latih sebanyak 3640 sampel dan jumlah sample pada data test yaitu 910 sampel dari total jumlah sample pada dataset yaitu 4550 sampel.

### Data Normalization
Normalisasi data digunakan agar model dapat bekerja lebih optimal karena model tidak perlu mengolah data dengan angka besar. Normalisasi biasanya mentransformasi data dalam skala tertentu. Untuk proyek ini kita akan normalisasi data 0 hingga 1 menggunakan MinMaxScaler.

### Menampilkan jumlah baris dan kolom apa saja yang ada, akan terlihat ada selisih jumlah sebesar 112 record (5291-5179). lalu kita coba pastikan lagi dengan melakukan penjumlahan data yang kosong saja
"""

df.info()

df.isnull().sum()

"""### Jika dilihat dari list kolom, ada kolom Date yang type datanya object tetapi berisi tanggal yang seharusnya typenya adalah datetime agar dapat di lakukan proses lebih lanjut."""

datetime_columns = ["Date"]
for column in datetime_columns:df[column] = pd.to_datetime(df[column])

"""### lakukan pengecekan kembali untuk type data Date apakah sudah berubah menjadi datetime"""

df.info()

"""# **Exploratory Data Analysis**

**Deskripsi Variabel**

* Date : Tanggal pencatatan Data
* Open : Harga buka dihitung perhari
* High : Harga tertinggi perhari
* Low : Harga terendah perhari
* Close : Harga tutup dihitung perhari
* Adj Close : Harga penutupan pada hari tersebut setelah disesuaikan dengan aksi korporasi seperti right issue, stock split atau stock reverse.
* Volume : Volume transaksi

### Pengulangan untuk mengecek kolom yang isinya null dengan fungsi simpleimputer untuk mengisi nilai yang hilang dalam data dengan nilai konstan atau menggunakan statistik dari kolom tempat nilai yang hilang berada. Statistik yang digunakan dapat berupa rata-rata, median, atau paling sering. Lalu lakukan pengecekan kembali apakah masih ada data yg null atau tidak
"""

col_missing = [col for col in df.columns if df[col].isnull().any()]

imputer = SimpleImputer()
df[col_missing] = imputer.fit_transform(df[col_missing])
df.head()

df.isnull().sum()

"""# **Explore Statistic Information**

masing-masing kolom memiliki informasi, antara lain:

* **count** adalah jumlah sampel pada data.
* **mean** adalah nilai rata-rata.
* **std** adalah standar deviasi.
* **min** adalah nilai minimum.
* **25%** adalah kuartil pertama.
* **50%** adalah kuartil kedua (nilai tengah).
* **75%** adalah kuartil ketiga.
* **max** adalah nilai maksimum
"""

df.describe()

"""# **Data visualiation**

Memvisualisasikan data menggunakan boxplot untuk semua fitur numerik:
"""

numerical_col = [col for col in df.columns if df[col].dtypes == 'float64']
plt.subplots(figsize=(10,7))
sns.boxplot(data=df[numerical_col]).set_title("Gold Price")
plt.show()
#sns.boxplot(x=df['Volume'])

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3-Q1
df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]

df.shape

numerical_col = [col for col in df.columns if df[col].dtypes == 'float64']
plt.subplots(figsize=(10,7))
sns.boxplot(data=df[numerical_col]).set_title("Gold Price")
plt.show()

"""# **Univariate Analysis**

Fitur yang diprediksi pada kasus ini adalah terfokus pada 'Adj Close'
"""

cols = 3
rows = 2
fig = plt.figure(figsize=(cols * 5, rows * 5))

for i, col in enumerate(numerical_col):
  ax = fig.add_subplot(rows, cols, i + 1)
  sns.histplot(x=df[col], bins=30, kde=True, ax=ax)
fig.tight_layout()
plt.show()

"""# **Multivariate Analysis**

Selanjutnya kita akan menganalisis korelasi fitur "Adj Close" terhadap fitur lain seperti "Open", "High", "Low", "Close" dan "Volume". Dapat disimpulkan bahwa "Adj Close" memiliki korelasi positif yang kuat terhadap "Open", "High", "Low" dan "Close", sedangkan untuk fitur "Volume" memiliki korelasi sedang terhadap fitur "Adj Close"


"""

sns.pairplot(df[numerical_col], diag_kind='kde')
plt.show()

plt.figure(figsize=(15,8))
corr = df[numerical_col].corr().round(2)
sns.heatmap(data=corr, annot=True, vmin=-1, vmax=1, cmap='coolwarm', linewidth=1)
plt.title('Correlation matrix for numerical feature', size=15)
plt.show()

df = df.drop(['Date','Volume', 'Close'], axis=1)
df.head()

"""# **Splitting Dataset**"""

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#melatih data menjadi data train dan data test 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

print(len(X_train), 'records')
print(len(y_train), 'records')
print(len(X_test), 'records')
print(len(y_test), 'records')

"""# **Data Normalization**

Untuk melakukan normalisasi data kita akan menggunakan library MinMaxScaler. Fungsi normalisasi pada data agar model lebih cepat dalam mempelajari data karena data telah diubah pada rentang tertentu seperti antara 0 dan 1
"""

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = pd.DataFrame(columns=['train_mse', 'test_mse'], index=['SVR', 'KNN', 'GradientBoosting'])

"""# **Modeling**

Hyperparameter Tuning adalah proses mencari nilai optimal untuk hyperparameter suatu model dalam machine learning atau deep learning. Hyperparameter adalah variabel konfigurasi eksternal yang digunakan ilmuwan data untuk mengelola pelatihan model machine learning. Tujuan dari Hyperparameter Tuning adalah untuk meningkatkan kinerja model dengan memilih kombinasi hyperparameter yang optimal.
"""

def grid_search(model, hyperparameters):
  results = GridSearchCV(
      model,        # Model machine learning yang ingin dioptimalkan.
      hyperparameters,
      cv=5,         # Jumlah fold untuk cross-validation
      verbose=1,    # Tingkat kecerahan output
      n_jobs=6      # Jumlah prosesor yang digunakan untuk komputasi
  )

  return results

"""Fitting pada model SVR,Gradient boost dan KNN dengan menggunakan data pelatihan"""

svr = SVR()
hyperparameters = {
    'kernel': ['rbf'],
    'C': [0.001, 0.01, 0.1, 10, 100, 1000],
    'gamma': [0.3, 0.03, 0.003, 0.0003]
}

svr_search = grid_search(svr, hyperparameters)
svr_search.fit(X_train, y_train)

gradient_boost = GradientBoostingRegressor()
hyperparameters = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'n_estimators': [250, 500, 750, 1000],
    'criterion': ['friedman_mse', 'squared_error']
}

gradient_boost_search = grid_search(gradient_boost, hyperparameters)
gradient_boost_search.fit(X_train, y_train)

knn = KNeighborsRegressor()
hyperparameters = {
    'n_neighbors': range(1, 10)
}

knn_search = grid_search(knn, hyperparameters)
knn_search.fit(X_train, y_train)

"""# **Model Training**"""

# Cetak hasil SVR model
print("Best Parameters: ", svr_search.best_params_)
print("Best Score: ", svr_search.best_score_)

# Cetak hasil gradient_boost model
print("Best Parameters: ", gradient_boost_search.best_params_)
print("Best Score: ", gradient_boost_search.best_score_)

# Cetak hasil knn model
print("Best Parameters: ", knn_search.best_params_)
print("Best Score: ", knn_search.best_score_)

svr = SVR(C=10, gamma=0.3, kernel='rbf')
svr.fit(X_train, y_train)

gradient_boost = GradientBoostingRegressor(criterion='squared_error', learning_rate=0.01, n_estimators=1000)
gradient_boost.fit(X_train, y_train)

knn = KNeighborsRegressor(n_neighbors=9)
knn.fit(X_train, y_train)

"""# **Model Evaluation**

Membuat dictionary yang berisi 3 nama model yang sudah kita buat permodelannya diatas, lalu dengan pengulangan digunakan untuk menghitung Mean Squared Error (MSE) untuk setiap modelnya.
Hasilnya dapat dilihat bahwa GradientBoosting memiliki nilai MSE yang lebih kecil dibandingkan dengan SVR dan KNN, yang menunjukkan bahwa model lebih akurat.
Lalu untuk lebih jelaskan kita bisa visualisasikan dengan bentuk graph
"""

model_dict = {
    'SVR': svr,
    'GradientBoosting': gradient_boost,
    'KNN': knn,
}

for name, model in model_dict.items():
  models.loc[name, 'train_mse'] = mean_squared_error(y_train, model.predict(X_train))
  models.loc[name, 'test_mse'] = mean_squared_error(y_test, model.predict(X_test))

models.head()

models.sort_values(by='test_mse', ascending=False).plot(kind='bar', zorder=3)

svr_accuracy = svr.score(X_test, y_test)*100
knn_accuracy = knn.score(X_test, y_test)*100
gb_accuracy = gradient_boost.score(X_test, y_test)*100

list_evaluasi = [[svr_accuracy],
            [knn_accuracy],
            [gb_accuracy]]
evaluasi = pd.DataFrame(list_evaluasi,
                        columns=['Accuracy (%)'],
                        index=['SVR','K-Nearest Neighbor', 'Gradient Boost'])
evaluasi

# Membuat model regresi
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Menghitung R2
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Membuat tabel hasil
results = {
    "Data Split": ["Train", "Test"],
    "R² Value": [r2_train, r2_test],
}
results_df = pd.DataFrame(results)

# Menampilkan tabel hasil
print(results_df)

"""Dari hasil evaluasi di atas dengan menggunakan MSE dan R2 atau coefficient of determination dapat memberikan informasi bahwa ketiga model yang dibangun memiliki performa di atas 99% mendekati 100%. Dimana dapat dilihat juga bahwa model dengan algoritma KNN memiliki performa yang diukur dengan nilai akurasi yang lebih baik dari dua model lainnya yaitu model dengan algoritma SVR dan Gradient Boost.
Sehingga untuk dapat memprediksi harga emas selama 30 hari kedepan bisa menggunakan algoritma KNN karena secara akurasinya hampir mendekati sempurna 100%
"""

X_30=X[-30:]
forecast=knn.predict(X_30)

df1=pd.DataFrame(forecast,columns=['Forecast'])
df1 = pd.concat([df, df1], ignore_index=True) #ignore_index=True used to reset index
df1.drop(['High', 'Low', 'Open'],axis=1,inplace=True)

df1.tail(35)

"""Berikut adalah nilai prediksi 30 hari kedepan yang didapat dari metode terbaik yaitu KNN yang dievaluasikan sebelumnya

## **Kesimpulan**

Dari hasil perbandingan dengan menggunakan 3 model diatas didapatkan hasil yang terbaik untuk bisa melakukan analisa prediksi harga emas dengan menggunakan KNN karena dari evaluasinya akurasinya paling tinggi hampir mendekati angka 100%

* Bagaimana menganalisa data harga *Emas*?

Tujuan proyek ini :
* Kita bisa memprediksi harga *Emas* dengan menggunakan model machine learning terbaik.
* Dengan didapatkannya prediksi harga emas untuk periode mendatang dapat  Membantu *trader* untuk dapat mengambil keputusan dalam melakukan pembelian pada *Emas*.

Langkah yang sudah dilakukan agar dapat mencapai tujuan proyek ini adalah sebagai berikut :
* Melakukan analisa, eksplorasi, pemrosesan pada data sampai dengan  memvisualisasikan data agar mendapatkan gambaran datanya tersebut.
* Menangani *missing value* pada data.
* Mencari korelasi pada data untuk mencari *dependant variable* dan *independent variable*.
* Menangani outlier pada data dengan menggunakan Metode IQR.
* Melakukan normalisasi pada data terutama pada fitur numerik.
* Membuat model regresi untuk memprediksi bilangan kontinu untuk memprediksi harga yang akan datang.    
* Menggunakan algoritma untuk perbandingan :
    * Support Vector Machine (Support Vector Regression)
    * K-Nearest Neighbors
    * Boosting Algorithm (Gradient Boosting Regression)
* Melakukan hyperparameter tuning agar model dapat berjalan pada performa terbaik dengan menggunakan teknik Grid Search
"""