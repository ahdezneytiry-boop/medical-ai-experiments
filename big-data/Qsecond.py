import pandas as pd
import numpy as np

date_rng = pd.date_range(start='2023-01-01', periods=10, freq='D')
df = pd.DataFrame({
    'Date': date_rng,
    'Value': np.random.randint(1, 100, size=10)
})
df['Rolling_3D'] = df.set_index('Date')['Value'].rolling('3D').mean().values
print(df)