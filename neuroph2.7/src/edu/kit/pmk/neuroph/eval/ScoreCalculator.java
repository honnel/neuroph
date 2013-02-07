package edu.kit.pmk.neuroph.eval;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

import edu.kit.pmk.neuroph.log.Log;
import edu.kit.pmk.neuroph.parallel.ILearner;

public class ScoreCalculator {

	public static Score[] trainAndCalculateOnPermutedSet(DataSet dataSet,
			double trainingSetRatio, int runs, ILearner... learners) {
		Score[] scores = new Score[learners.length];
		for (int l = 0; l < learners.length; l++) {
			scores[l] = new Score(runs, learners[l]);
		}

		for (int i = 0; i < runs; i++) {
			TestAndTrainingSet tats = TestAndTrainingSet.splitSetAndPermute(
					dataSet, trainingSetRatio);
			
			for (int l = 0; l < learners.length; l++) {
				ILearner currentLearner = learners[l];
				Log.info(currentLearner.getDescription(), String.format("%d Threads", currentLearner.getNumberOfThreads()));
				currentLearner.resetToUnlearnedState();
				long t0 = System.currentTimeMillis();
				currentLearner.learn(tats.getTrainingSet());
				long time = System.currentTimeMillis() - t0;
				double error = calculateError(currentLearner, tats.getTestSet());
				scores[l].times[i] = time;
				scores[l].errors[i] = error;
			}
		}
		return scores;
	}

	public static Score[] trainAndCalculateOnGivenTraingAndTestSet(
			DataSet trainingSet, DataSet testSet, ILearner... learners) {
		Score[] scores = new Score[learners.length];
		for (int l = 0; l < learners.length; l++) {
			scores[l] = new Score(1, learners[l]);
		}

		for (int l = 0; l < learners.length; l++) {
			learners[l].resetToUnlearnedState();
			long t0 = System.currentTimeMillis();
			learners[l].learn(trainingSet);
			long time = System.currentTimeMillis() - t0;
			double error = calculateError(learners[l], testSet);
			scores[l].times[0] = time;
			scores[l].errors[0] = error;
		}

		return scores;
	}

	private static double calculateError(ILearner learner, DataSet testSet) {
		double error = 0;
		NeuralNetwork neuralNet = learner.getNeuralNetwork();
		for (Neuron n : neuralNet.getOutputNeurons()) {
			n.setError(0.0);
		}
		for (DataSetRow testSetRow : testSet.getRows()) {
			neuralNet.setInput(testSetRow.getInput());
			neuralNet.calculate();
			Neuron[] outputNeurons = neuralNet.getOutputNeurons();
			for (int out = 0; out < outputNeurons.length; out++) {
				error += Math.abs(testSetRow.getDesiredOutput()[out]
						- outputNeurons[out].getOutput());
			}
		}
		return error / testSet.size();
	}
}
