# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
# Date:01-09-2025
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/content/jee_mains_2013_to_2025_top_30_ranks.csv")

data["Year"] = data["Year"].astype(int)
data["Rank"] = data["Rank"].astype(int)
data["Total_Marks"] = data["Total_Marks"].astype(float)

avg_rank_per_year = data.groupby("Year")["Rank"].mean().reset_index()
years = avg_rank_per_year["Year"].tolist()
ranks = avg_rank_per_year["Rank"].tolist()

X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, ranks)]
n = len(years)

b = (n * sum(xy) - sum(ranks) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(ranks) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, ranks)]

coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(ranks), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

print(f"\nLinear Trend: y={a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

trend_df = pd.DataFrame({
    "Year": years,
    "Average_Rank": ranks,
    "Linear Trend": linear_trend,
    "Polynomial Trend": poly_trend
}).set_index("Year")
#A - LINEAR TREND ESTIMATION

plt.figure(figsize=(8, 5))
trend_df["Average_Rank"].plot(color="blue", marker="o", label="Average Top 30 Rank")
trend_df["Linear Trend"].plot(color="black", linestyle="--", marker="o", label="Linear Trend")
plt.xlabel("Year")
plt.ylabel("Average Rank (Top 30 Students)")
plt.title("JEE Mains Top 30 - Linear Trend of Ranks")
plt.legend()
plt.show()
#B- POLYNOMIAL TREND ESTIMATION

plt.figure(figsize=(8, 5))
trend_df["Average_Rank"].plot(color="blue", marker="o", label="Average Top 30 Rank")
trend_df["Polynomial Trend"].plot(color="red", marker="o", label="Polynomial Trend")
plt.xlabel("Year")
plt.ylabel("Average Rank (Top 30 Students)")
plt.title("JEE Mains Top 30 - Polynomial Trend of Ranks")
plt.legend()
plt.show()
```
### OUTPUT
A - LINEAR TREND ESTIMATION

<img width="687" height="470" alt="21" src="https://github.com/user-attachments/assets/99c380c1-497e-42b8-a3ee-3aa2ae662909" />

B- POLYNOMIAL TREND ESTIMATION

<img width="687" height="470" alt="22" src="https://github.com/user-attachments/assets/9ec642c4-1f36-4e73-ba21-cbfb2039cd08" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
