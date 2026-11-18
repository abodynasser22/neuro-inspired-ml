import time
import numpy as np
from scipy import spatial
#import scipy


def get_random_word_vector(dimension, k):
    positions = np.random.permutation(dimension)[0:2 * k]
    minus_positions = positions[0:k]
    plus_positions = positions[k:2 * k]
    v = np.array(np.zeros(dimension),np.int8)
    v[minus_positions] = -1
    v[plus_positions] = 1
    return v

def get_random_word_vector_bin(dimension):
    k = (int)(dimension/2)
    positions = np.random.permutation(dimension)[0:2 * k]
    minus_positions = positions[0:k]
    plus_positions = positions[k:2 * k]
    v = np.array(np.zeros(dimension), np.int8)
    v[minus_positions] = -1
    v[plus_positions] = 1
    return v

def get_answer(letter, word,  values):
    answer_list = ['A', 'B', 'C', 'D']
    distance_function = cosine_norm
    min_value = max(distance_function(word, values[0]), distance_function(word, values[1]), distance_function(word, values[2]),
                    distance_function(word, values[3]))
    i = 0
    while min_value != distance_function(word,values[i]):
        i += 1
    if answer_list[i] == letter:
        return 1
    else:
        return 0

def get_answer_mod(words):
    distance_function = cosine_norm
    min_value = max(distance_function(words[0], words[1]), distance_function(words[0], words[2]), distance_function(words[0], words[3]),
                    distance_function(words[0], words[4]))
    i = 1
    # while i <=4 and min_value != distance_function(words[0],words[i]):
    #     i+=1
    if min_value == distance_function(words[0],words[1]):
        return 1
    else:
        return 0

def euclidian_norm(x,y):
    return np.linalg.norm(x-y)

def cosine_norm(x, y):
    return 1-spatial.distance.cosine(x,y)

def trunc_to_eight(words):
    i = 0
    for word in words:
        if len(word) > 8:
            words[i] = word[0:8]
        i += 1
    return words

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