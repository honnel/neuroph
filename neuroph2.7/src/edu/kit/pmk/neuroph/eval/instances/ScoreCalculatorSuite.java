package edu.kit.pmk.neuroph.eval.instances;

import org.neuroph.core.learning.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BatchParallelMomentumBackpropagation;
import org.neuroph.nnet.learning.LMS;

import edu.kit.pmk.neuroph.eval.Score;
import edu.kit.pmk.neuroph.eval.ScoreCalculator;
import edu.kit.pmk.neuroph.parallel.NeuralNetworkWrapper;
import edu.kit.pmk.neuroph.parallel.networkclones.ClonebasedConcurrentLearner;
import edu.kit.pmk.neuroph.parallel.networkclones.interpolation.NeuralNetInterpolatorType;

public class ScoreCalculatorSuite {

	public static void main(String[] args) {
		ScoreCalculatorSuite scs = new ScoreCalculatorSuite();
		scs.irisSample(4, 10, 3, 0.5);
		// scs.trainOneElementSet();
		// scs.testParallelBatchLearningRule();
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

	private void irisSample(int numThreads, int syncFrequency, int runs,
			double trainingSetRatio) {
		String inputFileName = org.neuroph.samples.IrisClassificationSample.class
				.getResource("data/iris_data_normalised.txt").getFile();
		// create training set from file
		DataSet irisDataSet = DataSet.createFromFile(inputFileName, 4, 3, ",");

		// create MultiLayerPerceptron neural network
		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(4, 300, 3);
		// neuralNet.setLearningRule(new MomentumBackpropagation());
		// ((LMS) neuralNet.getLearningRule()).setBatchMode(true);
		((LMS) neuralNet.getLearningRule()).setMaxIterations(60);

		ClonebasedConcurrentLearner ccl_Arith = new ClonebasedConcurrentLearner(
				numThreads, syncFrequency,
				NeuralNetInterpolatorType.ArithmeticMean, neuralNet);
		ClonebasedConcurrentLearner ccl_Min = new ClonebasedConcurrentLearner(
				numThreads, syncFrequency, NeuralNetInterpolatorType.Minimum,
				neuralNet);
		ClonebasedConcurrentLearner ccl_Max = new ClonebasedConcurrentLearner(
				numThreads, syncFrequency, NeuralNetInterpolatorType.Maximum,
				neuralNet);
		ClonebasedConcurrentLearner ccl_Geom = new ClonebasedConcurrentLearner(
				numThreads, syncFrequency,
				NeuralNetInterpolatorType.GeometricMean, neuralNet);
		NeuralNetworkWrapper mlp = new NeuralNetworkWrapper(neuralNet, 1);

		Score[] scores = ScoreCalculator.trainAndCalculateOnPermutedSet(
				irisDataSet, trainingSetRatio, runs, ccl_Arith, ccl_Min,
				ccl_Max, ccl_Geom, mlp);

		System.out.println("Clonebased_Arith: " + scores[0]);
		System.out.println("Clonebased_Min: " + scores[1]);
		System.out.println("Clonebased_Max: " + scores[2]);
		System.out.println("Clonebased_Geom: " + scores[3]);
		System.out.println("Sequential MLP: " + scores[4]);
	}

}
