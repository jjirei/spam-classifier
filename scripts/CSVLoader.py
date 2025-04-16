import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['length'] = df['message'].apply(len)

print (df.head())
print (df.isnull().sum())
print (df['label'].value_counts())

sns.countplot(x='label', data=df)
sns.histplot(data=df, x='length', hue='label', bins=50, kde=True)
plt.title("Spam vs Ham Distribution")
plt.show()