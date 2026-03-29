import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style for better visuals
sns.set(style="whitegrid")

# -------------------------------
# Load final dataset
# -------------------------------
df = pd.read_csv("data/final_dataset.csv")
df.columns = df.columns.str.strip()

print("Columns:", df.columns)


# -------------------------------
# 1. Temperature vs Height 🔥
# -------------------------------
plt.figure()
sns.scatterplot(x='ALTM', y='TMPC', hue='Storm_actual', data=df)

plt.title("Temperature vs Height")
plt.xlabel("Height (ALTM)")
plt.ylabel("Temperature (°C)")
plt.show()


# -------------------------------
# 2. Humidity Distribution 🔥
# -------------------------------
plt.figure()
sns.histplot(df['RELH'], kde=True)

plt.title("Humidity Distribution")
plt.xlabel("Relative Humidity (%)")
plt.ylabel("Frequency")
plt.show()


# -------------------------------
# 3. Humidity vs Storm (Box Plot)
# -------------------------------
plt.figure()
sns.boxplot(x='Storm_actual', y='RELH', data=df)

plt.title("Humidity vs Storm Condition")
plt.xlabel("Storm (0 = No, 1 = Yes)")
plt.ylabel("Humidity (%)")
plt.show()


# -------------------------------
# 4. Temperature vs Storm (Trend)
# -------------------------------
plt.figure()
sns.boxplot(x='Storm_actual', y='TMPC', data=df)

plt.title("Temperature vs Storm Condition")
plt.xlabel("Storm")
plt.ylabel("Temperature (°C)")
plt.show()