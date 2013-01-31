package edu.kit.pmk.neuroph.eval.experiment;

import org.neuroph.core.learning.DataSet;

import edu.kit.pmk.neuroph.parallel.ILearner;

public class Experiment {

	private final String DATASET_VALUE_DELIMITER = ",";

	ExperimentConfiguration myconfig;
	private int runs;
	private int max_iteration;
	private int hidden_neurons;
	private DataSet dataset;
	private double training_to_test_ratio;
	private ILearner learners;

	public Experiment(String identifier, String experimentConfigurationFile) {
		this.myconfig = new ExperimentConfiguration(identifier,
				experimentConfigurationFile);
		prepareExperiment(myconfig);
	}

	private void prepareExperiment(ExperimentConfiguration config) {
		runs = Integer.parseInt(config
				.getArgument(ExperimentConfigurationArgument.runs));
		max_iteration = Integer.parseInt(config
				.getArgument(ExperimentConfigurationArgument.max_iteration));
		hidden_neurons = Integer.parseInt(config
				.getArgument(ExperimentConfigurationArgument.hidden_neurons));
		int input_neurons = Integer.parseInt(config
				.getArgument(ExperimentConfigurationArgument.input_neurons));
		int output_neurons = Integer.parseInt(config
				.getArgument(ExperimentConfigurationArgument.output_neurons));
		String datasetFile = config
				.getArgument(ExperimentConfigurationArgument.dataset);
		training_to_test_ratio = Double
				.parseDouble(config
						.getArgument(ExperimentConfigurationArgument.training_to_test_ratio));
		dataset = DataSet.createFromFile(datasetFile, input_neurons,
				output_neurons, DATASET_VALUE_DELIMITER);
		learners = parseLearners(config);

	}

	private ILearner parseLearners(ExperimentConfiguration config2) {
		String[] learnerNames = myconfig.getArgument(ExperimentConfigurationArgument.learners).split(",");
		return null;
	}

}
