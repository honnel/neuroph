package edu.kit.pmk.neuroph.eval.instances;

import java.io.IOException;
import java.util.Arrays;

import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.learning.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.ParallelMultiLayerPerceptron;
import org.neuroph.nnet.learning.BatchParallelBackpropagation;
import org.neuroph.nnet.learning.LMS;

import edu.kit.pmk.neuroph.eval.Score;
import edu.kit.pmk.neuroph.eval.ScoreCalculator;
import edu.kit.pmk.neuroph.log.Log;
import edu.kit.pmk.neuroph.parallel.ILearner;
import edu.kit.pmk.neuroph.parallel.NeuralNetworkWrapper;
import edu.kit.pmk.neuroph.parallel.networkclones.ClonebasedConcurrentLearner;
import edu.kit.pmk.neuroph.parallel.networkclones.interpolation.NeuralNetInterpolatorType;
import edu.kit.pmk.neuroph.parallel.networkclones.revised.ClonebasedConcurrentLearnerRevised;

public class ScoreCalculatorSuite {

	private static final double TRAININGSET_RATIO = 0.5;
	private static final int SYNC_FREQUENCY = 100;
	private static final int COUNT_THREADS = 2;
	private static final int MAX_ITERATION = 2;
	private static final int COUNT_RUNS = 30;
	private static final int HIDDEN_NEURON_COUNT = 100;

	public static void main(String[] args) throws IOException {
		Log.setVerbose(true);
		ScoreCalculatorSuite scs = new ScoreCalculatorSuite();
		String inputFileName = org.neuroph.samples.IrisClassificationSample.class
				.getResource("data/iris_data_normalised.txt").getFile();
		// create training set from file
		// DataSet irisDataSet = DataSet.createFromFile(inputFileName, 4, 3,
		// ",");

		// CernFormatConverter cfc = new CernFormatConverter(
		// "data/cern/alldata/result.txt",
		// "data/cern/alldata/eventsPassSelection");
		// String outputFile = "data/cern/converted.txt";
		// cfc.writeToFile(outputFile);

		DataSet cernDataSet = DataSet.createFromFile(CERN_100,
				CERN_FULL_INPUTCOUNT, 1, ",");
		scs.compareDefaultToFinegrained(cernDataSet, HIDDEN_NEURON_COUNT, MAX_ITERATION, COUNT_THREADS, COUNT_RUNS, TRAININGSET_RATIO);
		scs.testClonebasedAndRevisedAndNormalMLP(cernDataSet,
				HIDDEN_NEURON_COUNT, MAX_ITERATION, COUNT_THREADS,
				SYNC_FREQUENCY, COUNT_RUNS, TRAININGSET_RATIO, NeuralNetInterpolatorType.Genetic);
		//
		// scs.testClonebasedAndNormalMLP(cernDataSet, HIDDEN_NEURON_COUNT,
		// MAX_ITERATION, COUNT_THREADS, SYNC_FREQUENCY, COUNT_RUNS,
		// TRAININGSET_RATIO, NeuralNetInterp8olatorType.ArithmeticMean);
		//
		// scs.testParallelBatchLearningRule(cernDataSet, HIDDEN_NEURON_COUNT,
		// MAX_ITERATION, COUNT_THREADS, COUNT_RUNS, TRAININGSET_RATIO);

		System.out.println("Ich teste liebend gerne neuronale Netze.");
	}

	private static final String CERN_FULL = "data/cern/15000rows.txt";
	private static final String CERN_EXAMPLE = "data/cern/example.txt";
	private static final String CERN_100 = "data/cern/100rows.txt";
	private static final String CERN_1000 = "data/cern/1000rows.txt";
	private static final String CERN_5000 = "data/cern/5000rows.txt";
	private static int CERN_FULL_INPUTCOUNT = 2853;
	private static int CERN_1000_INPUTCOUNT = 2853;
	private static int CERN_100_INPUTCOUNT = 2853;
	private static int CERN_EXAMPLE_INPUTCOUNT = 1278;

	private void testClonebasedAndRevisedAndNormalMLP(DataSet dataSet,
			int hiddenNeuronCount, int maxIterations, int numThreads,
			int syncFrequency, int runs, double trainingSetRatio,
			NeuralNetInterpolatorType interpolatorType) {

		System.out.println("+-+-+-+-+ Test Configuration +-+-+-+-+");
		System.out.println("Clonebased, Clonebased Revised, Seq MLP: "
				+ hiddenNeuronCount);
		System.out.println("Hidden Neurons: " + hiddenNeuronCount);
		System.out.println("Maximum Learning Iterations: " + maxIterations);
		System.out.println("#Threads: " + numThreads);
		System.out
				.println("For Unrevised: Interpolate clones after every <#> rows: "
						+ syncFrequency);
		System.out.println("Interpolation Type: ArithmeticMean");
		System.out.println("TraingSet to TestSet ratio: " + trainingSetRatio);
		System.out.println("Number of Test Runs: " + runs);
		System.out.println();

		// create MultiLayerPerceptron neural network
		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(
				dataSet.getInputSize(), hiddenNeuronCount,
				dataSet.getOutputSize());
		// neuralNet.setLearningRule(new MomentumBackpropagation());
		// ((LMS) neuralNet.getLearningRule()).setBatchMode(true);
		((LMS) neuralNet.getLearningRule()).setMaxIterations(maxIterations);

		ILearner[] learners = new ILearner[2];

		int i = 0;
		learners[0] = new ClonebasedConcurrentLearnerRevised(numThreads,
				maxIterations,interpolatorType,
				neuralNet, "Clonebased-Revised " + interpolatorType.name());
		// learners[1] = new ClonebasedConcurrentLearner(numThreads,
		// syncFrequency, NeuralNetInterpolatorType.ArithmeticMean,
		// neuralNet, "Clonebased ArithmeticMean");
		learners[1] = new NeuralNetworkWrapper(neuralNet, 1, "Sequential MLP");

		Score[] scores = ScoreCalculator.trainAndCalculateOnPermutedSet(
				dataSet, trainingSetRatio, runs, learners);

		System.out.println("\n+-+-+-+-+ Scores +-+-+-+-+");
		i = 0;
		for (ILearner l : learners) {
			System.out.println(l.getDescription() + ": " + scores[i++]);
		}
	}

	private void testClonebasedAndNormalMLP(DataSet dataSet,
			int hiddenNeuronCount, int maxIterations, int numThreads,
			int syncFrequency, int runs, double trainingSetRatio,
			NeuralNetInterpolatorType... types) {

		System.out.println("+-+-+-+-+ Test Configuration +-+-+-+-+");
		System.out.println("Hidden Neurons: " + hiddenNeuronCount);
		System.out.println("Maximum Learning Iterations: " + maxIterations);
		System.out.println("#Threads: " + numThreads);
		System.out.println("Interpolate clones after every <#> rows: "
				+ syncFrequency);
		System.out.println("Interpolation Types: " + Arrays.toString(types));
		System.out.println("TraingSet to TestSet ratio: " + trainingSetRatio);
		System.out.println("Number of Test Runs: " + runs);
		System.out.println();

		// create MultiLayerPerceptron neural network
		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(
				dataSet.getInputSize(), hiddenNeuronCount,
				dataSet.getOutputSize());
		// neuralNet.setLearningRule(new MomentumBackpropagation());
		// ((LMS) neuralNet.getLearningRule()).setBatchMode(true);
		((LMS) neuralNet.getLearningRule()).setMaxIterations(maxIterations);

		ILearner[] learners = new ILearner[types.length + 1];

		int i = 0;
		for (NeuralNetInterpolatorType type : types) {
			learners[i++] = new ClonebasedConcurrentLearner(numThreads,
					syncFrequency, type, neuralNet);
		}
		learners[i] = new NeuralNetworkWrapper(neuralNet, 1);

		Score[] scores = ScoreCalculator.trainAndCalculateOnPermutedSet(
				dataSet, trainingSetRatio, runs, learners);

		System.out.println("\n+-+-+-+-+ Scores +-+-+-+-+");
		i = 0;
		for (NeuralNetInterpolatorType type : types) {
			System.out.println("Clonebased-" + type + ": " + scores[i++]);
		}
		System.out.println("Sequential MLP: " + scores[i]);
	}

	private void testParallelBatchLearningRule(DataSet dataSet,
			int hiddenNeuronCount, int maxIterations, int numThreads, int runs,
			double trainingSetRatio) {
		System.out.println("+-+-+-+-+ Test Configuration +-+-+-+-+");
		System.out.println("Hidden Neurons: " + hiddenNeuronCount);
		System.out.println("Maximum Learning Iterations: " + maxIterations);
		System.out.println("#Threads: " + numThreads);
		System.out.println("TraingSet to TestSet ratio: " + trainingSetRatio);
		System.out.println("Number of Test Runs: " + runs);
		System.out.println();

		MultiLayerPerceptron parNeuralNet = new MultiLayerPerceptron(
				dataSet.getInputSize(), hiddenNeuronCount, 1);
		parNeuralNet.setLearningRule(new BatchParallelBackpropagation(
				numThreads));
		((LMS) parNeuralNet.getLearningRule()).setMaxIterations(maxIterations);
		NeuralNetworkWrapper parBatch = new NeuralNetworkWrapper(parNeuralNet,
				numThreads, "Parallel MLP BatchBackpropagation");

		MultiLayerPerceptron seqNeuralNet = new MultiLayerPerceptron(
				dataSet.getInputSize(), hiddenNeuronCount, 1);
		((LMS) seqNeuralNet.getLearningRule()).setBatchMode(true);
		((LMS) seqNeuralNet.getLearningRule()).setMaxIterations(maxIterations);
		NeuralNetworkWrapper seqBatch = new NeuralNetworkWrapper(seqNeuralNet,
				1, "Sequential MLP BatchBackpropagation");

		Score[] scores = ScoreCalculator.trainAndCalculateOnPermutedSet(
				dataSet, trainingSetRatio, runs, parBatch, seqBatch);

		System.out.println("\n+-+-+-+-+ Scores +-+-+-+-+");
		System.out.println(parBatch.getDescription() + ": " + scores[0]);
		System.out.println(seqBatch.getDescription() + ": " + scores[1]);
	}

	private void trainOneElementSet() {
		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(4, 300, 3);
		((LMS) neuralNet.getLearningRule()).setMaxIterations(10);

		DataSet oneElementSet = new DataSet(4, 3);
		oneElementSet.addRow(new double[] { 0.5, 0.7, 0.3, 0.1 }, new double[] {
				0.4, .2, 1 });
		neuralNet.learn(oneElementSet);
		System.out.println("I just did terminate.");
	}

	private void compareDefaultToFinegrained(DataSet dataSet, int hiddenNeuronCount, int maxIterations, int numThreads, int runs, double trainingSetRatio) {

		System.out.println("+-+-+-+-+ Test Configuration +-+-+-+-+");
		System.out.println("Fine grained vs Seq MLP: " + hiddenNeuronCount);
		System.out.println("Hidden Neurons: " + hiddenNeuronCount);
		System.out.println("Maximum Learning Iterations: " + maxIterations);
		System.out.println("#Threads: " + numThreads);
		System.out.println("Interpolation Type: ArithmeticMean");
		System.out.println("TraingSet to TestSet ratio: " + trainingSetRatio);
		System.out.println("Number of Test Runs: " + runs);
		System.out.println();

		MultiLayerPerceptron parallelNet = new ParallelMultiLayerPerceptron(numThreads, dataSet.getInputSize(), hiddenNeuronCount, dataSet.getOutputSize());

		// create MultiLayerPerceptron neural network
		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(dataSet.getInputSize(), hiddenNeuronCount, dataSet.getOutputSize());
		// neuralNet.setLearningRule(new MomentumBackpropagation());
		// ((LMS) neuralNet.getLearningRule()).setBatchMode(true);
		((LMS) neuralNet.getLearningRule()).setMaxIterations(maxIterations);

		ILearner[] learners = new ILearner[2];

		for (int layerIdx = 0; layerIdx < neuralNet.getLayersCount(); layerIdx++) {
			Layer l = neuralNet.getLayerAt(layerIdx);
			for (int neuronIdx = 0; neuronIdx < l.getNeuronsCount(); neuronIdx++) {
				Neuron neuron = l.getNeuronAt(neuronIdx);
				for (int weightIdx = 0; weightIdx < neuron.getWeights().length; weightIdx++) {
					parallelNet.getLayerAt(layerIdx).getNeuronAt(neuronIdx).getWeights()[weightIdx].value = neuron.getWeights()[neuronIdx].value;
					parallelNet.getLayerAt(layerIdx).getNeuronAt(neuronIdx).getWeights()[weightIdx].weightChange = neuron.getWeights()[neuronIdx].weightChange;
				}
			}
		}

		learners[0] = new NeuralNetworkWrapper(parallelNet, 8, "Parallel MLP");
		// learners[1] = new ClonebasedConcurrentLearner(numThreads,
		// syncFrequency, NeuralNetInterpolatorType.ArithmeticMean,
		// neuralNet, "Clonebased ArithmeticMean");
		learners[1] = new NeuralNetworkWrapper(neuralNet, 1, "Sequential MLP");

		Score[] scores = ScoreCalculator.trainAndCalculateOnPermutedSet(dataSet, trainingSetRatio, runs, learners);

		System.out.println("\n+-+-+-+-+ Scores +-+-+-+-+");
		int i = 0;
		for (ILearner l : learners) {
			System.out.println(l.getDescription() + ": " + scores[i++]);
		}
	}
}
