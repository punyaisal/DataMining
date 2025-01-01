# Data
x = [5, 4, 6, 5, 5, 5, 6, 6, 2, 7, 7]  # Usia mobil (tahun)
y = [85, 103, 70, 82, 89, 98, 66, 95, 169, 70, 48]  # Harga mobil ($100)

# Panjang data
n = len(x)

# Hitung sigma x, sigma y, sigma xy, sigma x^2
sum_x = sum(x)
sum_y = sum(y)
sum_xy = sum([xi * yi for xi, yi in zip(x, y)])
sum_x2 = sum([xi**2 for xi in x])

# Hitung slope (m) dan intercept (b)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
b = (sum_y - m * sum_x) / n

# Cetak hasil
print(f"Persamaan regresi: y = {m:.2f}x + {b:.2f}")
