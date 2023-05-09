import pandas as pd
import sklearn.preprocessing as pre
from sklearn.feature_selection import *
from sklearn.model_selection import *
from sklearn.linear_model import *
import matplotlib.pyplot as plt
import seaborn as sns

s = "-"*50

main_data = pd.read_csv(
    r"D:\SomeCode\AI_Programing\valo\Players.csv", encoding="utf-8")

# main_data = pd.read_csv(
#     r"/mnt/d/SomeCode/AI_Programing/valo/Players.csv", encoding="utf-8")

shape = main_data.shape
dtype = main_data.dtypes
columns = main_data.columns

print(s, end="\n")
print(dtype)
print(s, end="\n")
print(shape)
print(s, end="\n")
print(columns)
print(s, end="\n")

print(s, end="\n")
print(f"The number of duplicated rows is => {main_data.duplicated().sum()}")
print(s, end="\n")
print(f"The empty cells are at ==> \n{main_data.isna().sum()}")
print(s, end="\n")

print(main_data["Earnings"])

# main_data cleaning

main_data["Earnings"] = list(
    map(lambda x: x.replace("$", "").replace(",", ""), main_data["Earnings"]))
main_data["Earnings"] = pd.to_numeric(main_data["Earnings"])


dtype = main_data.dtypes

print(s, end="\n")
print(dtype)
print(s, end="\n")
# ----------------------------------------------------------------

##################################################
# Note:-                                         #
#       the target label is the price,           #
# and the features are the rest and i will       #
# split the features to calclate the corrlation  #
# betwen the fetures and the lable i will use    #
# r_regration                                    #
##################################################

for i in range(shape[1]):
    if dtype[i] == "object":
        pro = pre.LabelEncoder()
        main_data[columns[i]] = pro.fit_transform(main_data[columns[i]])
        # print(s)
        # print(pro.classes_)

scaler = pre.MinMaxScaler()
main_data = scaler.fit_transform(main_data)

# corr_data = pd.DataFrame(main_data, columns=columns)

# r = corr_data.corr()
# sns.heatmap(r, annot=True)
# plt.show()


features = main_data[:, 1:-1]

lable = main_data[:, -1]

r = r_regression(features, lable)

print(s)
print(r)

best = SelectKBest(r_regression, k=3)
x_best = best.fit_transform(features, lable)

print(s)
print("the best three features are \n", best.get_support())
print(s)
print(x_best)
print(s)

x_train, x_test, y_train, y_test = train_test_split(
    x_best, lable, test_size=0.5)

model = LinearRegression()

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(s)
print(f"the model acc is ==> {acc}")
