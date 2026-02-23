import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_test, y_pred):
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

def plot_results(cost_history, y_test, y_pred):
    plt.figure()
    plt.plot(cost_history)
    plt.title("Cost Convergence")
    plt.savefig("reports/cost_convergence.png")

    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Delivery Time")
    plt.ylabel("Predicted Delivery Time")
    plt.title("Actual vs Predicted")
    plt.savefig("reports/prediction_plot.png")