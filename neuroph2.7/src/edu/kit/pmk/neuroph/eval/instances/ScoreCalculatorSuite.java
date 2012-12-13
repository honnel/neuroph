package edu.kit.pmk.neuroph.eval.instances;

import org.neuroph.core.learning.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.LMS;

import edu.kit.pmk.neuroph.eval.Score;
import edu.kit.pmk.neuroph.eval.ScoreCalculator;
import edu.kit.pmk.neuroph.parallel.NeuralNetworkWrapper;
import edu.kit.pmk.neuroph.parallel.networkclones.ClonebasedConcurrentLearner;
import edu.kit.pmk.neuroph.parallel.networksiblings.SiblingBasedConcurrentLearner;

public class ScoreCalculatorSuite {

	public static void main(String[] args) {
		ScoreCalculatorSuite scs = new ScoreCalculatorSuite();
		scs.irisSample(2, 8, 30, 0.5);
	}

	private void irisSample(int numThreads, int syncFrequency, int runs,
			double trainingSetRatio) {
		String inputFileName = org.neuroph.samples.IrisClassificationSample.class
				.getResource("data/iris_data_normalised.txt").getFile();
		// create training set from file
		DataSet irisDataSet = DataSet.createFromFile(inputFileName, 4, 3, ",");

		// create MultiLayerPerceptron neural network
		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(4, 300, 3);
		((LMS) neuralNet.getLearningRule()).setMaxIterations(60);

		ClonebasedConcurrentLearner ccl = new ClonebasedConcurrentLearner(
				numThreads, syncFrequency, neuralNet);
		SiblingBasedConcurrentLearner sbl = new SiblingBasedConcurrentLearner(
				numThreads, syncFrequency, neuralNet);
		NeuralNetworkWrapper mlp = new NeuralNetworkWrapper(neuralNet, 1);
		
		
		NeuralNetworkWrapper parallelBatchRule = new NeuralNetworkWrapper(neuralNet, 2);

		Score[] scores = ScoreCalculator.trainAndCalculateOnPermutedSet(
				irisDataSet, trainingSetRatio, runs, ccl, sbl, mlp);

		System.out.println("Clonebased: " + scores[0]);
		System.out.println("Siblingbased: " + scores[1]);
		System.out.println("Sequential MLP: " + scores[2]);
	}

}
