import matplotlib.pyplot as plt

# Data
algorithms = ['Alexnet46', 'VGG', 'GoogleNet', 'Resnet', 'Proposed System']
accuracies = [88-55, 68-65, 88-82, 74-72, 95-75]

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(algorithms, accuracies, color='orange')

# Formatting
plt.title('Accuracy Comparison of Different Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy Range (%)')
plt.ylim([0, 35])
plt.grid(True)

# Displaying
plt.show()
