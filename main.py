import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

import datasets
import utils
import simple_classifier
import knn
import perceptron
import dt


def analysis_part1():
    print("Part 1: Loading and Processing Data...")
    # Fetch raw data
    print("Fetching MNIST data (this might take a few seconds)...")
    X_raw, y_raw = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )

    # Process the data using your implemented function in utils
    X_train, X_test, y_train, y_test = utils.process_data(X_raw, y_raw)

    # Q1: Plot a 3 and an 8
    # TODO: Find one example of a +1 (an '8') and one example of a -1 (a '3') in X_train.
    # Use utils.plot_images() to display them side-by-side.
    eight_index = 0
    three_index = 0

    for i in range(len(y_train)):
        if y_train[i] == 1.0:
            eight_index = i
        if y_train[i] == -1.0:
            three_index = i

    utils.plot_images(
        X_train[eight_index], "Plot of 8 (+1)", X_train[three_index], "Plot of 3 (-1)"
    )

    # Q2: Print the shapes of the training and testing sets.
    # TODO: Print the shape of X_train, y_train, X_test, and y_test.
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Q3: Print the number of positive (+1) and negative (-1) examples in both sets.
    # TODO: Count and print how many +1s and -1s are in y_train and y_test.
    sum_train_positive = 0
    sum_train_negative = 0
    sum_test_positive = 0
    sum_test_negative = 0

    for i in y_train:
        if i == 1.0:
            sum_train_positive += 1
        elif i == -1.0:
            sum_train_negative += 1

    for i in y_test:
        if i == 1.0:
            sum_test_positive += 1
        elif i == -1.0:
            sum_test_negative += 1

    print("Training set: +1 labels =", sum_train_positive)
    print("Training set: -1 labels =", sum_train_negative)
    print("Testing set: +1 labels =", sum_test_positive)
    print("Testing set: -1 labels =", sum_test_negative)

    # Q4: Most Frequent Baseline Evaluation
    clf = simple_classifier.MostFrequentClassClassifier()
    clf.train(X_train, y_train)
    preds = clf.predict(X_test)

    # TODO: Compute and print the test accuracy of the MostFrequentClassClassifier.
    # Hint: Use utils.compute_accuracy(y_test, preds)
    print("MostFrequentClassClassifier test accuracy:", utils.compute_accuracy(y_test, preds))


def analysis_part2():
    print("\nPart 2 (KNN on 10% of CIFAR-10):")

    # Fetch CIFAR-10 data
    print("Fetching CIFAR-10 dataset...")
    X_raw_cifar, y_raw_cifar = utils.fetch_cifar10()

    # Process the data
    X_train_c, X_test_c, y_train_c, y_test_c = utils.process_cifar_data(
        X_raw_cifar, y_raw_cifar
    )

    # Q5: Visualizing Neighbors for a Correct Prediction
    # TODO: Train KNN(k=5).
    # Find a test example where the prediction is CORRECT.
    # Get its 5 nearest neighbors from the training set.
    # Plot using utils.plot_image_and_neighbors.
    clf = knn.KNN(k=5)
    clf.train(X_train_c, y_train_c)
    preds = clf.predict(X_test_c)

    correct_i = None
    for i in range(len(y_test_c)):
        if preds[i] == y_test_c[i]:
            correct_i = i
            break

    correct_img = X_test_c[correct_i]
    neighbor_indices = clf.get_k_neighbors_indices(correct_img)

    utils.plot_image_and_neighbors(correct_img, neighbor_imgs=X_train_c[neighbor_indices])

    # Q6: Visualizing Neighbors for a Mistake
    # TODO: Find a test example where the prediction is WRONG.
    # Get its 5 nearest neighbors from the training set.
    # Plot using utils.plot_image_and_neighbors.

    wrong_i = None
    for i in range(len(y_test_c)):
        if preds[i] != y_test_c[i]:
            wrong_i = i
            break

    wrong_img = X_test_c[wrong_i]
    neighbor_indices = clf.get_k_neighbors_indices(wrong_img)

    utils.plot_image_and_neighbors(wrong_img, neighbor_imgs=X_train_c[neighbor_indices])

    # Q7: Hyperparameters, Overfitting, and Underfitting
    k_vals = [3, 5, 7, 9, 11, 13]
    train_accs = []
    test_accs = []
    # TODO: Loop over k, train the model, get Train Acc and Test Acc.
    # Plot Train and Test accuracies vs. k.
    # Hint: plotting code below
    for k in k_vals:
        clf = knn.KNN(k=k)
        clf.train(X_train_c, y_train_c)
        train_preds = clf.predict(X_train_c)
        test_preds = clf.predict(X_test_c)
        train_acc = utils.compute_accuracy(y_train_c, train_preds)
        test_acc = utils.compute_accuracy(y_test_c, test_preds)

        train_accs.append(train_acc)
        test_accs.append(test_acc)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, train_accs, marker="o", label="Train Accuracy")
    plt.plot(k_vals, test_accs, marker="s", label="Test Accuracy")
    plt.xlabel("K (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs. K (Airplane vs Frog)")
    plt.legend()
    plt.grid(True)
    plt.show()


def analysis_part3():
    print("\nPart 3 (Perceptron):")

    # Q9 & Q10: Blob Dataset Analysis
    print("Fetching blob data...")
    X_train_blob, X_test_blob, y_train_blob, y_test_blob = utils.get_blob_data()

    # TODO: Train a Perceptron for 50 epochs on the blob training data.
    # TODO: Compute and print the final training and testing accuracies.
    # TODO: Use utils.plot_decision_boundary to visualize the model on the blob data.
    p = perceptron.Perceptron(num_epochs=50)
    p.train(X_train_blob, y_train_blob)

    train_preds = p.predict(X_train_blob)
    test_preds = p.predict(X_test_blob)

    train_acc = utils.compute_accuracy(y_train_blob, train_preds)
    test_acc = utils.compute_accuracy(y_test_blob, test_preds)
    print("Blob training accuracy:", train_acc)
    print("Blob testing accuracy:", test_acc)

    utils.plot_decision_boundary(X_train_blob, y_train_blob, p)

    # Q11: Collinear Blobs Problem
    print("\nFetching collinear data...")
    X_coll, y_coll = utils.get_collinear_blobs()

    # TODO: Train a Perceptron for 100 epochs on the collinear data.
    # TODO: Print the final training accuracy.
    # TODO: Use utils.plot_decision_boundary to visualize the model on the collinear data.
    p_coll = perceptron.Perceptron(num_epochs=100)
    p_coll.train(X_coll, y_coll)

    train_preds_coll = p_coll.predict(X_coll)
    train_acc_coll = utils.compute_accuracy(y_coll, train_preds_coll)

    print("Collinear training accuracy:", train_acc_coll)

    utils.plot_decision_boundary(X_coll, y_coll, p_coll)


def analysis_part4():
    print("\n--- Analysis Part 4 (Decision Trees) ---")

    # We use datasets.py for explainable features instead of MNIST/CIFAR
    tennis_X, tennis_y = datasets.TennisData.X, datasets.TennisData.Y  # train set
    tennis_Xte, tennis_yte = (
        datasets.TennisData.Xte,
        datasets.TennisData.Yte,
    )  # test set

    sentiment_X, sentiment_y = datasets.SentimentData.X, datasets.SentimentData.Y
    sentiment_Xte, sentiment_yte = (
        datasets.SentimentData.Xte,
        datasets.SentimentData.Yte,
    )

    # Q13: Evaluate performance with depths 1, 3, and 5 on SentimentData
    # TODO: Train DT with max_depth 1, 3, and 5 on sentiment_X/y. Evaluate and print accuracy.
    for depth in [1, 3, 5]:
        tree = dt.DT({"max_depth": depth})
        tree.train(sentiment_X, sentiment_y)

        train_preds = tree.predict(sentiment_X)
        test_preds = tree.predict(sentiment_Xte)

        train_acc = utils.compute_accuracy(sentiment_y, train_preds)
        test_acc = utils.compute_accuracy(sentiment_yte, test_preds)

        print(f"Training accuracy {train_acc}, test accuracy {test_acc}")

    # Q14: Learning Curves (Dataset Size)
    # TODO: Generate learning curves by changing the dataset size (e.g., using SentimentData).
    # Hint: use plotting code from above, you may also make it a function and call it from `utils`
    sizes = [1, 5, 10, 20, 50, 100, 200, 500, datasets.SentimentData.X.shape[0]]
    train_accs = []
    test_accs = []

    for i in sizes:
        X_small = sentiment_X[:i]
        y_small = sentiment_y[:i]

        tree = dt.DT({"max_depth": 5})
        tree.train(X_small, y_small)
        train_preds = tree.predict(X_small)
        test_preds = tree.predict(sentiment_Xte)
        train_acc = utils.compute_accuracy(y_small, train_preds)
        test_acc = utils.compute_accuracy(sentiment_yte, test_preds)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"N = {i}")
        print(f"Training accuracy {train_acc}, test accuracy {test_acc}")

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_accs, marker="o", label="Training Accuracy")
    plt.plot(sizes, test_accs, marker="s", label="Test Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Accuracy vs. Dataset Size")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Q15: Hyperparameter Curves (Max Depth)
    # TODO: Generate hyperparameter curves by varying the max_depth hyperparameter on SentimentData.
    depths = [1, 3, 5, 7, 11, 15, 20, 30]
    train_accs = []
    test_accs = []

    for depth in depths:
        tree = dt.DT({"max_depth": depth})
        tree.train(sentiment_X, sentiment_y)

        train_preds = tree.predict(sentiment_X)
        test_preds = tree.predict(sentiment_Xte)
        train_acc = utils.compute_accuracy(sentiment_y, train_preds)
        test_acc = utils.compute_accuracy(sentiment_yte, test_preds)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"depth = {depth}")
        print(f"Training accuracy {train_acc}, test accuracy {test_acc}")

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_accs, marker="o", label="Training Accuracy")
    plt.plot(depths, test_accs, marker="s", label="Test Accuracy")
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Accuracy vs. Tree Depth")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # You can comment/uncomment these out to run specific parts
    # analysis_part1()
    # analysis_part2()
    # analysis_part3()
    analysis_part4()
