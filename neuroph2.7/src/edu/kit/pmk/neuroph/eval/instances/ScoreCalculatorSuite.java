package edu.kit.pmk.neuroph.eval.instances;

import java.io.IOException;

import org.neuroph.core.learning.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BatchParallelMomentumBackpropagation;
import org.neuroph.nnet.learning.LMS;

import edu.kit.pmk.neuroph.eval.Score;
import edu.kit.pmk.neuroph.eval.ScoreCalculator;
import edu.kit.pmk.neuroph.parallel.ILearner;
import edu.kit.pmk.neuroph.parallel.NeuralNetworkWrapper;
import edu.kit.pmk.neuroph.parallel.networkclones.ClonebasedConcurrentLearner;
import edu.kit.pmk.neuroph.parallel.networkclones.interpolation.NeuralNetInterpolatorType;
import edu.kit.pmk.neuroph.samples.CernParticleCollision.CernFormatConverter;

public class ScoreCalculatorSuite {

	public static void main(String[] args) throws IOException {
		ScoreCalculatorSuite scs = new ScoreCalculatorSuite();
		String inputFileName = org.neuroph.samples.IrisClassificationSample.class
				.getResource("data/iris_data_normalised.txt").getFile();
		// create training set from file
		DataSet irisDataSet = DataSet.createFromFile(inputFileName, 4, 3, ",");

		CernFormatConverter cfc = new CernFormatConverter(
				"data/cern/result.txt", "data/cern/eventsPassSelectionExample");
		String outputFile = "data/cern/converted.txt";
		cfc.writeToFile(outputFile);
		DataSet cernDataSet = DataSet.createFromFile(outputFile,
				cfc.getInputCount(), cfc.getOutputCount(), ",");

		// scs.trainClonebasedAndNormalMLP(irisDataSet, 300, 4, 10, 3, 0.5,
		// NeuralNetInterpolatorType.ArithmeticMean,
		// NeuralNetInterpolatorType.Maximum);
		scs.trainClonebasedAndNormalMLP(cernDataSet, 1000, 4, 10, 1, 0.5,
				NeuralNetInterpolatorType.ArithmeticMean);
		// scs.trainOneElementSet();
		// scs.testParallelBatchLearningRule();
	}

	private void trainClonebasedAndNormalMLP(DataSet dataSet,
			int hiddenNeuronCount, int numThreads, int syncFrequency, int runs,
			double trainingSetRatio, NeuralNetInterpolatorType... types) {

		// create MultiLayerPerceptron neural network
		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(
				dataSet.getInputSize(), hiddenNeuronCount,
				dataSet.getOutputSize());
		// neuralNet.setLearningRule(new MomentumBackpropagation());
		// ((LMS) neuralNet.getLearningRule()).setBatchMode(true);
		((LMS) neuralNet.getLearningRule()).setMaxIterations(60);

		ILearner[] learners = new ILearner[types.length + 1];

		int i = 0;
		for (NeuralNetInterpolatorType type : types) {
			learners[i++] = new ClonebasedConcurrentLearner(numThreads,
					syncFrequency, type, neuralNet);
		}
		learners[i] = new NeuralNetworkWrapper(neuralNet, 1);

		Score[] scores = ScoreCalculator.trainAndCalculateOnPermutedSet(
				dataSet, trainingSetRatio, runs, learners);

		i = 0;
		for (NeuralNetInterpolatorType type : types) {
			System.out.println("Clonebased-" + type + ": " + scores[i++]);
		}
		System.out.println("Sequential MLP: " + scores[i]);
	}

	private void testParallelBatchLearningRule() {
		String inputFileName = org.neuroph.samples.IrisClassificationSample.class
				.getResource("data/iris_data_normalised.txt").getFile();
		// create training set from file
		DataSet irisDataSet = DataSet.createFromFile(inputFileName, 4, 3, ",");

		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(4, 300, 3);
		int threads = 2;
		neuralNet.setLearningRule(new BatchParallelMomentumBackpropagation(
				threads));
		NeuralNetworkWrapper wrap = new NeuralNetworkWrapper(neuralNet, threads);

		Score[] scores = ScoreCalculator.trainAndCalculateOnPermutedSet(
				irisDataSet, 0.5, 1, wrap);

		System.out.println("BatchParallelMomentumBP: " + scores[0]);
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

}
