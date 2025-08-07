import pickle

from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate


def main() -> None:
    """
    Load and visualize an Optuna study with various plots.
    
    This function loads a saved Optuna study from a pickle file and creates
    three different visualization plots: optimization history, parameter
    importances, and parallel coordinate plots.
    """
    # Load the study from pickle file
    with open("minigames/move_to_beacon/optuna/3/study.pkl", "rb") as f:
        study = pickle.load(f)

    # Create visualization plots
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig3 = plot_parallel_coordinate(study)

    # Display the plots
    fig1.show()
    fig2.show()
    fig3.show()

if __name__ == "__main__":
    main()
