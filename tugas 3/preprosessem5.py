import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = r'C:\kuliah\semester 5\Data mining\tugas 3\Data1.xlsx'
data = pd.read_excel(file_path)

# Mengisi nilai yang hilang menggunakan SimpleImputer
# Untuk kolom numerik, menggunakan nilai rata-rata (mean)
imputer = SimpleImputer(missing_values=pd.NA, strategy='mean')
data['Tinggi Badan'] = imputer.fit_transform(data[['Tinggi Badan']])

# Untuk kolom kategorikal, menggunakan nilai yang paling sering (modus)
imputer_categorical = SimpleImputer(strategy='most_frequent')
data['Pembayaran'] = imputer_categorical.fit_transform(data[['Pembayaran']]).ravel()

# Bersihkan kolom 'Pendapatan', hilangkan simbol $, koma, dan ubah ke numerik
data['Pendapatan'] = data['Pendapatan'].replace({r'\$': '', ',': ''}, regex=True).astype(float)

# Mengisi nilai NaN di kolom Pendapatan
data['Pendapatan'] = imputer.fit_transform(data[['Pendapatan']])

# Membulatkan kolom 'Tinggi Badan' menjadi 3 digit tanpa koma
data['Tinggi Badan'] = data['Tinggi Badan'].round().astype(int)

# Membuat kolom 'Pendapatan' menjadi 5 digit, mengalikan dengan 10000 jika perlu
# Jika data Pendapatan sudah dalam skala ribuan, pastikan tidak membagi lagi.
data['Pendapatan'] = (data['Pendapatan'] / 10000).round() * 10000

# Cek jika ada nilai yang masih 0
print("Pendapatan sebelum disimpan:", data['Pendapatan'].unique())

# Simpan hasilnya ke file Excel atau CSV
output_excel_path = r'C:\kuliah\semester 5\Data mining\tugas 3\Data1_cleaned.xlsx'
output_csv_path = r'C:\kuliah\semester 5\Data mining\tugas 3\Data1_cleaned.csv'

data.to_excel(output_excel_path, index=False)
data.to_csv(output_csv_path, index=False)

print("Data preprocessing selesai dan hasil disimpan ke Excel dan CSV.")
