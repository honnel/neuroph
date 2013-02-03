package edu.kit.pmk.neuroph.eval.experiment;

import java.io.IOException;
import java.util.Date;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BatchParallelBackpropagation;
import org.neuroph.nnet.learning.LMS;

import edu.kit.pmk.neuroph.eval.Score;
import edu.kit.pmk.neuroph.eval.ScoreCalculator;
import edu.kit.pmk.neuroph.log.Log;
import edu.kit.pmk.neuroph.parallel.ILearner;
import edu.kit.pmk.neuroph.parallel.NeuralNetworkWrapper;
import edu.kit.pmk.neuroph.parallel.networkclones.ClonebasedConcurrentLearner;
import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;
import edu.kit.pmk.neuroph.parallel.networkclones.interpolation.NeuralNetInterpolatorType;
import edu.kit.pmk.neuroph.parallel.networkclones.revised.ClonebasedConcurrentLearnerRevised;

public class Experiment {

	private final String DATASET_VALUE_DELIMITER = ",";

	ExperimentConfiguration myconfig;
	private int runs;
	private int threads;
	private int max_iteration;
	private double sync_frequency = 0.25;
	private int hidden_neurons;
	private DataSet dataset;
	private double training_to_test_ratio;
	private ILearner[] learners;

	public Experiment(String identifier, String experimentConfigurationFile) {
		this.myconfig = new ExperimentConfiguration(identifier,
				experimentConfigurationFile);
		prepareExperiment(myconfig);
		return;
	}

	private void prepareExperiment(ExperimentConfiguration config) {
		runs = Integer.parseInt(config
				.getArgument(ExperimentConfigurationArgument.runs));
		threads = Integer.parseInt(config
				.getArgument(ExperimentConfigurationArgument.threads));
		max_iteration = Integer.parseInt(config
				.getArgument(ExperimentConfigurationArgument.max_iteration));
		String syncf = config
				.getArgument(ExperimentConfigurationArgument.sync_frequency);
		if (syncf != null)
			sync_frequency = Double.parseDouble(syncf);
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
		learners = parseLearners(config
				.getArgument(ExperimentConfigurationArgument.learners));

	}

	// possible learner identifiers (keyword or class name)
	private final String[] MLP = { "mlp",
			MultiLayerPerceptron.class.getSimpleName().toLowerCase() };
	private final String[] BATCH = { "batch", "batch" };
	private final String[] BATCH_PARALLEL = { "batch_parallel",
			BatchParallelBackpropagation.class.getSimpleName().toLowerCase() };
	private final String[] CLONEBASED = { "clonebased",
			ClonebasedConcurrentLearner.class.getSimpleName().toLowerCase() };
	private final String[] CLONEBASED_REVISED = {
			"clonebased_revised",
			ClonebasedConcurrentLearnerRevised.class.getSimpleName()
					.toLowerCase() };

	private ILearner[] parseLearners(String learnerIDs) {
		String[] learnerNames = learnerIDs.split(",");
		this.learners = new ILearner[learnerNames.length];

		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(
				dataset.getInputSize(), hidden_neurons, dataset.getOutputSize());
		((LMS) neuralNet.getLearningRule()).setMaxIterations(max_iteration);
		for (int i = 0; i < learnerNames.length; i++) {
			String name = learnerNames[i].trim().toLowerCase();
			if (name.equals(MLP[0]) || name.equals(MLP[1])) {
				setupMLP(neuralNet, i);
			} else if (name.equals(BATCH[0]) || name.equals(BATCH[1])) {
				setupBatch(neuralNet, i);
			} else if (name.equals(BATCH_PARALLEL[0])
					|| name.equals(BATCH_PARALLEL[1])) {
				setupBatchParallel(neuralNet, i);
			} else {
				String[] clonebased_interpolatortype = name.split("-");
				if (clonebased_interpolatortype[0].equals(CLONEBASED[0])
						|| clonebased_interpolatortype[0].equals(CLONEBASED[1])) {
					setupClonebased(neuralNet, i,
							clonebased_interpolatortype[1]);
				} else if (clonebased_interpolatortype[0]
						.equals(CLONEBASED_REVISED[0])
						|| clonebased_interpolatortype[0]
								.equals(CLONEBASED_REVISED[1])) {
					setupClonebasedRevised(neuralNet, i,
							clonebased_interpolatortype[1]);
				} else {
					throw new IllegalArgumentException("Unkown Learner '"
							+ name + "'!");
				}
			}
		}
		return learners;
	}

	private void setupClonebased(MultiLayerPerceptron neuralNet, int pos,
			String interpolatorType) {
		NeuralNetInterpolatorType type = getInterpolatorType(interpolatorType);
		learners[pos] = new ClonebasedConcurrentLearner(threads,
				(int) (sync_frequency * dataset.size()), type, neuralNet,
				CLONEBASED[0] + "-" + type.name());
	}

	private void setupClonebasedRevised(MultiLayerPerceptron neuralNet,
			int pos, String interpolatorType) {
		NeuralNetInterpolatorType type = getInterpolatorType(interpolatorType);
		learners[pos] = new ClonebasedConcurrentLearnerRevised(threads,
				max_iteration, type, neuralNet, CLONEBASED_REVISED[0] + "-"
						+ type.name());
	}

	private NeuralNetInterpolatorType getInterpolatorType(String typestring) {
		for (NeuralNetInterpolatorType t : NeuralNetInterpolatorType.values()) {
			if (typestring.equalsIgnoreCase(t.name())) {
				return t;
			}
		}
		throw new IllegalArgumentException(
				"Unknown NeuralNetInterpolatorType '" + typestring + "'!");
	}

	private void setupBatchParallel(MultiLayerPerceptron neuralNet, int pos) {
		NeuralNetwork batchNet = null;
		try {
			batchNet = (NeuralNetwork) FastDeepCopy.createDeepCopy(neuralNet);
		} catch (ClassNotFoundException | IOException e) {
			e.printStackTrace();
		}
		batchNet.setLearningRule(new BatchParallelBackpropagation(threads));
		learners[pos] = new NeuralNetworkWrapper(batchNet, threads,
				BATCH_PARALLEL[0]);
	}

	private void setupBatch(MultiLayerPerceptron neuralNet, int pos) {
		NeuralNetwork batchNet = null;
		try {
			batchNet = (NeuralNetwork) FastDeepCopy.createDeepCopy(neuralNet);
		} catch (ClassNotFoundException | IOException e) {
			e.printStackTrace();
		}
		((LMS) batchNet.getLearningRule()).setBatchMode(true);
		learners[pos] = new NeuralNetworkWrapper(batchNet, threads, BATCH[0]);
	}

	private void setupMLP(MultiLayerPerceptron neuralNet, int pos) {
		learners[pos] = new NeuralNetworkWrapper(neuralNet, threads, MLP[0]);
	}

	public void doExperiment() {
		Log.resetLoggingInstance();
		Log.setVerbose(true);
		Log.info("ExperimentConfiguration", myconfig.toString());
		Score[] scores = ScoreCalculator.trainAndCalculateOnPermutedSet(
				dataset, training_to_test_ratio, runs, learners);
		for (int i = 0; i < scores.length; i++) {
			Log.logScore(learners[i].getDescription(), scores[i]);
		}
		try {
			Log.writeAsDebugLog(myconfig.getName() + "_results_["
					+ (new Date()).hashCode() + "]");
		} catch (IOException e) {
			System.err.println("Failed to write Log!");
			e.printStackTrace();
		}

	}

	public static void main(String[] args) {
		Experiment exp = new Experiment("testconfig",
				"../eval/cernData/experiment_configs/testconfig.txt");
		exp.doExperiment();
	}

}
