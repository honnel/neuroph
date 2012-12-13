package edu.kit.pmk.neuroph.eval;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

import edu.kit.pmk.neuroph.parallel.ILearner;

public class ScoreCalculator {

	public static Score[] trainAndCalculateOnPermutedSet(DataSet dataSet,
			double trainingSetRatio, int runs, ILearner... learners) {
		Score[] scores = new Score[learners.length];
		for (int l = 0; l < learners.length; l++) {
			scores[l] = new Score(0, 0, learners[l]);
		}

		for (int i = 0; i < runs; i++) {
			TestAndTrainingSet tats = TestAndTrainingSet.splitSetAndPermute(
					dataSet, trainingSetRatio);
			for (int l = 0; l < learners.length; l++) {
				learners[l].resetToUnlearnedState();
				long t0 = System.currentTimeMillis();
				learners[l].learn(tats.getTrainingSet());
				long time = System.currentTimeMillis() - t0;
				double error = calculateError(learners[l], tats.getTestSet());
				scores[l].time += time;
				scores[l].error += error;
			}
		}
		for (int l = 0; l < learners.length; l++) {
			Score sc = scores[l];
			sc.error /= runs;
		}
		return scores;
	}

	public static Score[] trainAndCalculateOnGivenTraingAndTestSet(
			DataSet trainingSet, DataSet testSet, ILearner... learners) {
		Score[] scores = new Score[learners.length];
		for (int l = 0; l < learners.length; l++) {
			scores[l] = new Score(0, 0, learners[l]);
		}

		for (int l = 0; l < learners.length; l++) {
			learners[l].resetToUnlearnedState();
			long t0 = System.currentTimeMillis();
			learners[l].learn(trainingSet);
			long time = System.currentTimeMillis() - t0;
			double error = calculateError(learners[l], testSet);
			scores[l].time = time;
			scores[l].error = error;
		}

		return scores;
	}

	private static double calculateError(ILearner learner, DataSet testSet) {
		double error = 0;
		NeuralNetwork neuralNet = learner.getNeuralNetwork();
		for (DataSetRow testSetRow : testSet.getRows()) {
			neuralNet.setInput(testSetRow.getInput());
			neuralNet.calculate();
			for (Neuron neuron : neuralNet.getOutputNeurons()) {
				error += neuron.getError();
			}
		}
		return error / testSet.size();
	}
}
