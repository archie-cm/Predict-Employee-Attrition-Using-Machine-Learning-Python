# Predict-Employee-Attrition-Using-Machine-Learning-Python

MINI PROJECT 5 - Improving Employee Retention by Predicting Employee Attrition Using Machine Learning
TUGAS  1
-- Handling Missing Value
Terdapat 6 kolom yg missing value
Kolom IkutProgramLOP memiliki null value> 50 % maka perlu di drop
Null value pada Kolom SkorKepuasanPegawai diisi modus
Null Value pada Kolom AlasanResign diisi masih_bekerja, krn null menandakan dia belum resign
Kolom JumlahKeikutsertaanProjek, JumlahKeterlambatanSebulanTerakhir, JumlahKetidakhadiran diisi median

-- Mengganti value yang tidak sesuai (Hint: Perhatikan kolom "PernahBekerja")
Value 'yes' direplace menjadi 1. Tidak perlu direplace pun bisa langsung di-drop karena kolom ini hanya terdiri dari 1 unique value.

TUGAS 2
-- Soal No 1,2,3
Step:
Buat kolom baru 'tahun_hiring' dr kolom 'TanggalHiring' --> ambil tahunnya saja
Buat kolom baru 'tahun_resign' dr kolom 'Tanggalresign' --> ambil tahunnya saja (perhatikan ada value anomali)
Buat df baru utk hiring (cth df_hire) yg merupakan hasil perhitungan jumlah pekerja yg di-hire setiap tahunnya. Gunakan EnterpriseID dan tahun_hiring dlm agregatnya
Buat df baru utk resign (cth df_resign) yg merupakan hasil perhitungan jumlah pekerja yg resign setiap tahunnya. Gunakan EnterpriseID dan tahun_resign dlm agregatnya
Join df_hire dan df_resign (Outer join), bisa dibuat sbg df baru lagi (cth df_join) --> utk kolom yg NaN bisa diisi 0 mnggkn fillna
Sesuaikan value yg masih bernilai 0 pada kolom Tahun_Hiring di df_join

-- 4. Hitung total karyawan yang resign, yang masih bertahan, dan perubahannya pada setiap tahunnya dari table hasil join pada tahap dua (2)
Buat kolom `total_karyawan` dari df_join dgn cara:  df_join['total_hired'].cumsum() - df_join[total_resigned] -> Ini artinya jumlah karyawan di setiap tahunnya.
Buat kolom 'perubahan' dari df_join dgn cara: df_join['total_hired'] - df_join[total_resigned]  --> Karyawan yg di-hire dikurang karyawan yg resign

-- Soal nomor 5 dan 6 Plot dan interpretasi
Gunakan waterfall chart sprti referensi yg sdh diberikan di LMS


TUGAS 3
-- Soal No 1 dan Soal No 2
Buat kolom baru `isResign` dari kolom TanggalResign dari df original --> Jika valuenya mengandung tanggal maka Yes, jika tdk Maka No
Buat df baru (df_not_resign) yg merupakan jumlah pekerja yg tidak resign di setiap bidang pekerjaan. Gunakan kolom EnterpriseID dan Pekerjaan dlm agregatnya dan jgn lupa filter yg tidak resign saja.
Buat df baru (df_resign) yg merupakan jumlah pekerja yg resign di setiap bidang pekerjaan. Gunakan kolom EnterpriseID dan Pekerjaan dlm agregatnya dan jgn lupa filter yg resign saja.
Outer join antara df_not_resign dan df_resign  dan jgn lupa isi NaN dgn nilai 0 mnggkn fillna(0) --> jadi df baru misal df_gabungan

-- Soal No 3 dan 4
Buat kolom baru 'total_employee' pada dataframe df_gabungan dgn cara: Existing_employee + Resigned_Employee (kedua kolom ini dapat dari hasill agregasi pada soal no 1 dan 2)
Buat kolom baru `Existing_Percentage`  pada dataframe df_gabungan dgn cara: Existing_employee / total_employee * 100
Buat Plot yang menampilkan persentase employee yang masih ada berdasarkan divisi pekerjaannya (cth bisa pake bar plot)

-- Soal No 5 dan 6 dan 7
filter data dari df_gabungan yg presentase resignnya paling tinggi. Stlh itu agregasi -> df_gabungan.groupby([yg diminta di soal apa sj]).count()["EnterpriseID"]
Plot dgn Sunburst Charts in Python
Setelah itu interpretasi


TUGAS 4
-- Soal No 1
a. Feature Engineering
Buat Fitur baru 'keikusertaanproject' dari kolom JumlahKeikutsertaanProjek --> Jika > 0 maka True dan lainnya False (mappingkan sbg 0 dan 1)
 Cek kembali fitur 'isResign' yg tlh dibuat pada Tugas 3 --> pastikan sdh dimappingkan sbg 0 dan 1
Buat fitur baru `lama_bekerja` dgn cara: kolom tahun_resign - tahun_hiring
Buat fitur baru `usia_hired` (usia saat di-hire perusahaan) dgn cara: kolom tahun_hiring - Tanggal_lahir (jgn lupa ambil tahun saja dari tgl lahir)
Buat fitur baru `jarak_penilaian_tahun` (jarak antara tnggal penilaian karyawan dan tanggal saat di-hire): kolom TanggalPenilaianKaryawan - Tahun_Hiring
Buat fitur baru `divisi` dari kolom Pekerjaan utk mengurangi jumlah unique value. Misal dibagi menjadi 3 kategori saja yaitu: divisi engineering, divisi data, dan divisi product.

b. Categorical Encoding
Sesuaikan value '-' pada kolom StatusPernikahan
Kolom yg perlu dilakukan Label encoding: TingkatPendidikan, JenjangKarirPerfomancePegawai
Kolom yg perlu dilakukan OHE: StatusPernikahan, StatusKepegawaian, Divisi (hasil feature engineering dari kolom pekerjaan), AsalDaerah
Frequency Encoding: HiringPlatform 

**Frequency encoding itu menghitung total value dari setiap kategori, lalu dibagi dgn total baris pada dataframe. Setelah itu, nilai tersebut digunakan utk mengganti kategori sebelumnya. Contoh: total rows pada dataframe = 287 dan jumlah value LinkedIn dari kolom HiringPlatform adalah 69, maka hasil dari 69/287 digunakan utk mengganti value LinkedIn. Begitupun value lain pada HiringPlatform.

c. Drop feature yg tdk di-butuhkan
DROP ["Username","JenisKelamin","NomorHP","Email","AlasanResign", "TanggalHiring","TanggalLahir","TanggalPenilaianKaryawan", "Tahun_Resign","TanggalResign","Tahun_Hiring"], dan fitur lainnya yg sdh dilakukan categorical encoding atau fetaure encoding

d. Cek data duplikat dan handling outlier

-- Soal No 2
Pisahkan feature dan target
Split menjadi data training dan data test
Bandingkan hasil Modelling dari beberapa teknik handling imbalance (RandomUnderSampler, TomekLinks, EditedNearestNeighbours, SMOTEENN, dan SMOTETomek)
Gunakan pipeline (ref https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)
-- Soal No 3
Lakukan modeling dari berbagai algoritma dgn teknik handling terbaik dari soal no 2
validasi pakai RepeatedStratifiedKFold
-- Soal No 4
Lakukan hyperparameter tuning dari model terbaik


TUGAS 5
Bisa gunakan shap values, feature importances, dan Plot partial dependence (ref: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.plot_partial_dependence.html, https://www.kaggle.com/code/dansbecker/partial-dependence-plots/notebook)
