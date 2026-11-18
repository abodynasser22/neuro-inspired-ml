from scipy import spatial
import time

def creating_text_word8(file1, file2):
    file = open(file1, 'r')
    new_file = open(file2, "w")
    for line in file:
        if line != "\n":
            words = line.split()
            words = trunc_to_eight(words)
            for word in words:
                new_file.write(word + " ")
            new_file.write("\n")

    file.close()
    new_file.close()

def new_tasa(file1, file2):
    maximum = 15000

    dictionary = {}
    text_file = open(file1, "r")
    for line in text_file:
        if line != "\n":
            words = line.split()
            words = trunc_to_eight(words)
            for word in words:
                if dictionary.get(word) is None:
                    dictionary[word] = 1
                else:
                    dictionary[word] += 1

    text_file.close()

    file = open(file1, 'r')
    new_file = open(file2, "w")
    for line in file:
        if line != "\n":
            words = line.split()
            words = trunc_to_eight(words)
            for word in words:
                if dictionary[word] < maximum:
                    new_file.write(word + " ")
                    # new_file.write("\n")

    file.close()
    new_file.close()

def trunc_to_eight(words):
    i = 0
    for word in words:
        if len(word) > 8:
            words[i] = word[0:8]
        i += 1
    return words

def get_answer_mod(words):
    distance_function = cosine_norm
    min_value = max(distance_function(words[0], words[1]), distance_function(words[0], words[2]), distance_function(words[0], words[3]),
                    distance_function(words[0], words[4]))
    i = 1
    while min_value != distance_function(words[0],words[i]):
        i += 1
    if i == 1:
        return 1
    else:
        return 0

def cosine_norm(x, y):
    return 1-spatial.distance.cosine(x,y)

def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def save_results_to_file(filename, results, total_training_time):
    total_training_time = 0.0
    with open(filename, 'w') as f:
        f.write("Final results:\n\n")
        f.write("Experiment start time: " + get_current_time() + "\n")
                
        for dimension, accuracies in results.items():
            avg_accuracy = sum(accuracies) / len(accuracies)
            std_dev = (sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5
            
            f.write(f"Dimension: {dimension}\n")
            f.write(f"  All runs: {[f'{acc:.2f}%' for acc in accuracies]}\n")
            f.write(f"  Average:  {avg_accuracy:.2f}%\n")
            f.write(f"  Std Dev:  {std_dev:.2f}%\n")
            f.write(f"  Min:      {min(accuracies):.2f}%\n")
            f.write(f"  Max:      {max(accuracies):.2f}%\n")

        f.write("Results analysis:\n")
        
        best_dimension = max(results, key=lambda dim: sum(results[dim]) / len(results[dim]))
        best_avg_accuracy = sum(results[best_dimension]) / len(results[best_dimension])
        f.write(f"Best performing dimension: {best_dimension} with average accuracy of {best_avg_accuracy:.2f}%\n")
        
    print(f"Results saved to {filename}")