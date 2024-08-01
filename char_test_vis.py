# First, create character-wise accuracy table in a CSV file by running ```char_test.py```
# Then visualize the result by running ```char_test_vis```

import pandas as pd
import matplotlib.pyplot as plt

# Read "Character-wise-accuracy.csv" with first row as header
# df = pd.read_csv("Character-acc_HRNetDBiLSTM.csv", header=0)
df = pd.read_csv("Character-wise-accuracy.csv", header=0)


check_char = ['ء', 'آ', 'أ', 'إ', 'ا', 'ب', 
              'ت', 'ة', 'ث', 'ج', 'ح', 'خ', 
              'د', 'ذ', 'ر', 'ز', 'س', 'ش', 
              'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 
              'ف', 'ق', 'ك', 'ل', 'م', 'ن', 
              'ه', 'و', 'ؤ', 'ي', 'ى', 'ئ']

# Plot the accuracy of each character in check_char in a bar chart and saves it
df[df["Alphabet"].isin(check_char)].plot.bar(x="Alphabet", y="Accuracy", rot=0)
# df[df["Accuracy"]>=50].plot.bar(x="Alphabet", y="Accuracy", rot=0)
plt.savefig("Character-wise-accuracy.png")