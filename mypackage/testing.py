import numpy as np
import matplotlib.pyplot as plt

X_test = np.random.rand(20, 3)
y_pred = np.random.rand(20)
N = y_pred.size

colorGroup = ['b','g','r','c','m','y','k','w']
plt.figure(1)

carat_pred = X_test[:,:1]

print(carat_pred)
print('\n')
print(y_pred)


plt.figure(1)

fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(X_test[:, 1:2])

results_file = r"C:\Users\Laurentiu\SVM_Results.txt"
file = open(results_file, "a")
file.write("\n")
file.write("================================================================")
file.close()
